#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nioh 3 G1M to glTF Converter

Parses Nioh 3 G1M files (G1MG v0x30303435) and converts them to glTF.
Handles Nioh 3 specific quirks:
  - POSITION stored as R16G16B16A16_UINT (decoded as UNORM mapped to bounding box)
  - SUBMESH metadata vertex/index counts are unreliable; uses full VB/IB buffers

Does NOT modify existing source files. Reuses parsing logic from g1m_export_meshes.py.

Usage:
    python nioh3_g1m_to_gltf.py <input_dir_or_g1m> [output_dir]
"""
#
# GitHub jiangnanyouzi/Nioh3-G1MS

import glob
import os
import io
import sys
import re
import copy
import json
import struct
import subprocess
import shutil
import numpy
from PIL import Image

# Ensure project root is on path so we can import existing modules
_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    from pyquaternion import Quaternion
    from g1m_export_meshes import (
        parseG1MG, parseG1MS, generate_fmts, generate_ib, generate_vb,
        calc_abs_skeleton, combine_skeleton, binary_oid_to_dict, name_bones, generate_vgmap,
        cull_vb
    )
    from lib_fmtibvb import write_vb_stream, write_ib_stream
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    sys.exit(1)

# If set to True, meshes will translate to the root node
translate_meshes = True

# Path to g1t_tool.py (from plan.md)
G1T_TOOL_PATH = r"D:\project\python\Nioh3-Model-Texture-Mapping-Database\g1t_tool.py"


def convert_format_for_gltf(dxgi_format):
    """Convert DXGI format string to glTF-compatible component info."""
    dxgi_format = dxgi_format.split('DXGI_FORMAT_')[-1]
    dxgi_format_split = dxgi_format.split('_')
    if len(dxgi_format_split) == 2:
        numtype = dxgi_format_split[1]
        vec_format = re.findall("[0-9]+", dxgi_format_split[0])
        vec_bits = int(vec_format[0])
        vec_elements = len(vec_format)
        if numtype in ['FLOAT', 'UNORM']:
            componentType = 5126
            dxgi_format = re.sub('[0-9]+', '32', dxgi_format)
            dxgi_format = re.sub('UNORM', 'FLOAT', dxgi_format)
            componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
        elif numtype == 'UINT':
            if vec_bits == 32:
                componentType = 5125
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 4
            elif vec_bits == 16:
                componentType = 5123
                componentStride = len(re.findall('[0-9]+', dxgi_format)) * 2
            elif vec_bits == 8:
                componentType = 5121
                componentStride = len(re.findall('[0-9]+', dxgi_format))
        accessor_types = ["SCALAR", "VEC2", "VEC3", "VEC4"]
        accessor_type = accessor_types[len(re.findall('[0-9]+', dxgi_format)) - 1]
        return {
            'format': dxgi_format,
            'componentType': componentType,
            'componentStride': componentStride,
            'accessor_type': accessor_type
        }
    else:
        return False


def convert_fmt_for_gltf(fmt):
    """Adapt a FMT dict for glTF output."""
    new_fmt = copy.deepcopy(fmt)
    stride = 0
    new_semantics = {'BLENDWEIGHT': 'WEIGHTS', 'BLENDINDICES': 'JOINTS'}
    need_index = ['WEIGHTS', 'JOINTS', 'COLOR', 'TEXCOORD']
    for i in range(len(fmt['elements'])):
        if new_fmt['elements'][i]['SemanticName'] in new_semantics.keys():
            new_fmt['elements'][i]['SemanticName'] = new_semantics[new_fmt['elements'][i]['SemanticName']]
        new_info = convert_format_for_gltf(fmt['elements'][i]['Format'])
        new_fmt['elements'][i]['Format'] = new_info['format']
        if new_fmt['elements'][i]['SemanticName'] in need_index:
            new_fmt['elements'][i]['SemanticName'] = new_fmt['elements'][i]['SemanticName'] + '_' + \
                new_fmt['elements'][i]['SemanticIndex']
        new_fmt['elements'][i]['AlignedByteOffset'] = stride
        new_fmt['elements'][i]['componentType'] = new_info['componentType']
        new_fmt['elements'][i]['componentStride'] = new_info['componentStride']
        new_fmt['elements'][i]['accessor_type'] = new_info['accessor_type']
        stride += new_info['componentStride']
    index_fmt = convert_format_for_gltf(fmt['format'])
    new_fmt['format'] = index_fmt['format']
    new_fmt['componentType'] = index_fmt['componentType']
    new_fmt['componentStride'] = index_fmt['componentStride']
    new_fmt['accessor_type'] = index_fmt['accessor_type']
    new_fmt['stride'] = stride
    return new_fmt


def convert_bones_to_single_file(submesh):
    """Convert G1M blendindices (count by 3) to standard indices."""
    bone_element_indices = [x for x in submesh['fmt']['elements'] if x['SemanticName'] == 'BLENDINDICES']
    if len(bone_element_indices) > 0:
        for i in range(len(bone_element_indices)):
            bone_element_index = int(bone_element_indices[i]['id'])
            for j in range(len(submesh['vb'][bone_element_index]['Buffer'])):
                for k in range(len(submesh['vb'][bone_element_index]['Buffer'][j])):
                    submesh['vb'][bone_element_index]['Buffer'][j][k] = \
                        int(submesh['vb'][bone_element_index]['Buffer'][j][k] // 3)
    return submesh


def list_of_utilized_bones(submesh, model_skel_data):
    """Return list of bone indices used by this submesh."""
    true_bone_map = {}
    if model_skel_data['jointCount'] > 1:
        for i in range(len(model_skel_data['boneList'])):
            true_bone_map[model_skel_data['boneList'][i]['bone_id']] = model_skel_data['boneList'][i]['i']
    return [true_bone_map[x] for x in submesh['vgmap'].keys()]


def fix_weight_groups(submesh):
    """Sanity-check and fix blend weight/index groups."""
    new_submesh = copy.deepcopy(submesh)
    blend_indices_idx = [i for i in range(len(new_submesh['fmt']['elements']))
                         if new_submesh['fmt']['elements'][i]['SemanticName'] == 'BLENDINDICES']
    blend_weights_idx = [i for i in range(len(new_submesh['fmt']['elements']))
                         if new_submesh['fmt']['elements'][i]['SemanticName'] in ['BLENDWEIGHT', 'BLENDWEIGHTS']]
    blidx_layers = dict(sorted({int(new_submesh['fmt']['elements'][i]['SemanticIndex']): i for i in blend_indices_idx}.items()))
    bl_wt_layers = dict(sorted({int(new_submesh['fmt']['elements'][i]['SemanticIndex']): i for i in blend_weights_idx}.items()))
    # Remove extra blendindices
    if len(blend_indices_idx) > len(blend_weights_idx):
        unknowns = [i for i in range(len(new_submesh['fmt']['elements'])) if new_submesh['fmt']['elements'][i]['SemanticName'] == 'UNKNOWN']
        j = len(unknowns)
        for i in blidx_layers:
            if i >= len(blend_weights_idx):
                new_submesh['fmt']['elements'][blidx_layers[i]]['SemanticName'] = 'UNKNOWN'
                new_submesh['fmt']['elements'][blidx_layers[i]]['SemanticIndex'] = str(j)
                j += 1
    if len(bl_wt_layers) > 0:
        max_layer = max(bl_wt_layers.keys())
        bone_element_index = blidx_layers[max_layer]
        weight_element_index = bl_wt_layers[max_layer]
        # Re-insert final missing weight group
        if len(new_submesh['vb'][bone_element_index]['Buffer'][0]) - len(new_submesh['vb'][weight_element_index]['Buffer'][0]) > 0:
            for _ in range(len(new_submesh['vb'][bone_element_index]['Buffer'][0]) - len(new_submesh['vb'][weight_element_index]['Buffer'][0])):
                for j in range(len(new_submesh['vb'][weight_element_index]['Buffer'])):
                    new_submesh['vb'][weight_element_index]['Buffer'][j].append(1 - sum(new_submesh['vb'][weight_element_index]['Buffer'][j]))
            prefices = ['R', 'G', 'B', 'A', 'D']
            weightformat = new_submesh['fmt']['elements'][weight_element_index]['Format']
            dxgi_format_split = weightformat.split('_')
            if len(dxgi_format_split) == 2:
                numtype = dxgi_format_split[1]
                vec_format = re.findall("[0-9]+", dxgi_format_split[0])
                vec_bits = int(vec_format[0])
                vec_elements = len(vec_format)
                vec_elements += 1
            new_submesh['fmt']['elements'][weight_element_index]['Format'] = \
                "".join(["{0}{1}".format(prefices[j], vec_bits) for j in range(vec_elements)]) + '_' + numtype
            new_submesh['fmt']['stride'] = str(int(int(new_submesh['fmt']['stride']) + vec_bits / 8))
            for j in range(weight_element_index + 1, len(new_submesh['fmt']['elements'])):
                new_submesh['fmt']['elements'][j]['AlignedByteOffset'] = \
                    str(int(int(new_submesh['fmt']['elements'][j]['AlignedByteOffset']) + vec_bits / 8))
        # Remove invalid small weights
        for i in range(len(new_submesh['vb'][weight_element_index]['Buffer'])):
            for j in range(len(new_submesh['vb'][weight_element_index]['Buffer'][i])):
                if new_submesh['vb'][weight_element_index]['Buffer'][i][j] < 0.00001:
                    new_submesh['vb'][weight_element_index]['Buffer'][i][j] = 0
        # Remove cloth weights from 4D meshes
        for i in range(len(new_submesh['vb'][weight_element_index]['Buffer'])):
            if not new_submesh['vb'][weight_element_index]['Buffer'][i][0] == max(new_submesh['vb'][weight_element_index]['Buffer'][i]):
                new_submesh['vb'][weight_element_index]['Buffer'][i] = [1] + [0 for x in new_submesh['vb'][weight_element_index]['Buffer'][i][1:]]
                new_submesh['vb'][bone_element_index]['Buffer'][i] = [0 for x in new_submesh['vb'][bone_element_index]['Buffer'][i]]
    return new_submesh


def fix_normal_type(submesh):
    normal_element_index = int([x for x in submesh['fmt']['elements'] if x['SemanticName'] == 'NORMAL'][0]['id'])
    if not submesh['fmt']['elements'][normal_element_index]['Format'] == 'R32G32B32_FLOAT':
        submesh['fmt']['elements'][normal_element_index]['Format'] = 'R32G32B32_FLOAT'
        submesh['vb'][normal_element_index]['Buffer'] = [submesh['vb'][normal_element_index]['Buffer'][i][0:3]
            for i in range(len(submesh['vb'][normal_element_index]['Buffer']))]
    return submesh


def fix_tangent_length(submesh):
    tangent_element_index = int([x for x in submesh['fmt']['elements'] if x['SemanticName'] == 'TANGENT'][0]['id'])
    for i in range(len(submesh['vb'][tangent_element_index]['Buffer'])):
        submesh['vb'][tangent_element_index]['Buffer'][i][0:3] = \
            (submesh['vb'][tangent_element_index]['Buffer'][i][0:3] / numpy.linalg.norm(submesh['vb'][tangent_element_index]['Buffer'][i][0:3])).tolist()
    return submesh


def decode_nioh3_positions(vb_element, fmt_elements, bbox):
    """
    Nioh 3 stores positions as R16G16B16A16_UINT.
    Decode as UNORM and map to the G1MG bounding box.
    Updates vb_element Buffer and Format in-place.
    Returns stride delta (new - old).
    """
    old_stride = 8  # R16G16B16A16_UINT
    new_stride = 12  # R32G32B32_FLOAT
    new_buffer = []
    for raw in vb_element['Buffer']:
        # raw is list of 4 uint16 values [x, y, z, w]
        x = bbox['min_x'] + (raw[0] / 65535.0) * (bbox['max_x'] - bbox['min_x'])
        y = bbox['min_y'] + (raw[1] / 65535.0) * (bbox['max_y'] - bbox['min_y'])
        z = bbox['min_z'] + (raw[2] / 65535.0) * (bbox['max_z'] - bbox['min_z'])
        new_buffer.append([x, y, z])
    vb_element['Buffer'] = new_buffer
    vb_element['Format'] = 'R32G32B32_FLOAT'
    # Update fmt element
    for elem in fmt_elements:
        if elem['SemanticName'] == 'POSITION':
            elem['Format'] = 'R32G32B32_FLOAT'
    return new_stride - old_stride


def _parse_g1mg_section(g1mg_stream, target_magic, e='<'):
    """Parse a single section from G1MG stream by magic."""
    with io.BytesIO(g1mg_stream) as f:
        f.read(4); f.read(8); f.read(4); f.read(28)
        section_count = struct.unpack(e + 'I', f.read(4))[0]
        for i in range(section_count):
            magic = struct.unpack(e + 'I', f.read(4))[0]
            size = struct.unpack(e + 'I', f.read(4))[0]
            count = struct.unpack(e + 'I', f.read(4))[0]
            data = f.read(size - 12)
            if magic == target_magic:
                return {'magic': magic, 'size': size, 'count': count, 'data': data}
    return None


def _decode_10bit_triangles(data_arr):
    """Decode packed 10-bit triangles from uint32 array."""
    return [[v & 0x3FF, (v >> 10) & 0x3FF, (v >> 20) & 0x3FF] for v in data_arr]


def _try_meshlet_path(subindex, g1mg_stream, model_mesh_metadata, fmts, e, bbox):
    """
    Attempt to build triangles from meshlet sections (0x1000b/0x1000c).
    Uses 0x10010 for meshlet-to-submesh assignment if available.
    Returns list of triangles, or None if meshlets not available / not applicable.
    """
    sec_b = _parse_g1mg_section(g1mg_stream, 0x0001000b, e)
    sec_c = _parse_g1mg_section(g1mg_stream, 0x0001000c, e)
    if not sec_b or not sec_c:
        return None

    # Parse 0x1000b
    b_data = sec_b['data']
    b_count, b_stride = struct.unpack(e + 'II', b_data[:8])
    b_arr = struct.unpack(e + 'I' * b_count, b_data[8:])

    # Parse 0x1000c
    c_data = sec_c['data']
    c_count, c_stride = struct.unpack(e + 'II', c_data[:8])
    c_entries = []
    for i in range(c_count):
        off = i * c_stride
        vals = struct.unpack(e + 'I' * (c_stride // 4), c_data[8 + off:8 + off + c_stride])
        c_entries.append(vals)

    # Parse 0x1000e BVH to identify leaf meshlets (byte-accurate: child idx at offset 16/20 == -1)
    leaf_meshlets = set(range(c_count))  # default: all if no BVH
    sec_e = _parse_g1mg_section(g1mg_stream, 0x0001000e, e)
    if sec_e:
        e_data = sec_e['data']
        if len(e_data) >= 8:
            e_count, e_stride = struct.unpack(e + 'II', e_data[:8])
            # Some models have fewer BVH nodes than meshlet entries (e.g. 295 vs 296).
            # Entries within e_count are checked against BVH; entries beyond have no node and are treated as leaf.
            if e_count <= c_count and e_stride >= 24:
                leaf_meshlets = set()
                for i in range(e_count):
                    off = 8 + i * e_stride
                    child0, child1 = struct.unpack(e + 'ii', e_data[off + 16:off + 24])
                    if child0 == -1 and child1 == -1:
                        leaf_meshlets.add(i)
                for i in range(e_count, c_count):
                    leaf_meshlets.add(i)

    # Parse 0x10010 for meshlet-to-submesh mapping
    sec_10 = _parse_g1mg_section(g1mg_stream, 0x00010010, e)
    meshlet_to_submesh = None
    if sec_10:
        data_10 = sec_10['data']
        if len(data_10) >= 8:
            count_10, stride_10 = struct.unpack(e + 'II', data_10[:8])
            if count_10 <= c_count and stride_10 >= 12:
                meshlet_to_submesh = {}
                for i in range(count_10):
                    off = 8 + i * stride_10
                    a, b, c_val = struct.unpack(e + 'III', data_10[off:off + 12])
                    submesh_id = (c_val >> 16) & 0xFFFF
                    meshlet_to_submesh[a] = submesh_id

    # Get IB data for this specific submesh
    subvbs = [x for x in model_mesh_metadata['sections'] if x['type'] == 'SUBMESH'][0]
    ib_section = [x for x in model_mesh_metadata['sections'] if x['type'] == 'INDEX_BUFFER'][0]
    sibindex = subvbs['data'][subindex]['indexBufferIndex']
    ib_entry = ib_section['data'][sibindex]
    with io.BytesIO(g1mg_stream) as f:
        f.seek(ib_entry['offset'])
        raw = f.read(ib_entry['stride'] * ib_entry['count'])
        if ib_entry['stride'] == 4:
            my_ib_data = list(struct.unpack(e + 'I' * ib_entry['count'], raw))
        elif ib_entry['stride'] == 2:
            my_ib_data = list(struct.unpack(e + 'H' * ib_entry['count'], raw))
        else:
            my_ib_data = []

    tris = []
    seen = set()
    for mi in range(len(c_entries)):
        # Skip internal BVH nodes; only render leaf meshlets
        if mi not in leaf_meshlets:
            continue

        # Use 0x10010 mapping if available; otherwise include all meshlets
        if meshlet_to_submesh is not None:
            if meshlet_to_submesh.get(mi) != subindex:
                continue

        c_off, c_cnt, u1, u2 = c_entries[mi]
        end = u1 + u2

        # Sanity check: u1 and u2 are relative to this submesh's IB
        if end > len(my_ib_data):
            continue

        local_tris = _decode_10bit_triangles(b_arr[c_off:c_off + c_cnt])
        remap = my_ib_data[u1:u1 + u2]
        for tri in local_tris:
            try:
                g_tri = [remap[idx] for idx in tri]
                tri_key = tuple(sorted(g_tri))
                if tri_key not in seen:
                    seen.add(tri_key)
                    tris.append(g_tri)
            except IndexError:
                pass

    return tris if tris else None


def generate_nioh3_submesh(subindex, g1mg_stream, model_mesh_metadata, fmts, e='<', bbox=None):
    """
    Generate submesh for Nioh 3 using the FULL corresponding VB/IB.
    The parsed SUBMESH vertex/index counts are unreliable in v0x30303435.
    If meshlet sections (0x1000b/0x1000c/0x1000e) are present, uses them
    for precise triangle extraction; otherwise falls back to strip parsing.
    """
    subvbs = [x for x in model_mesh_metadata['sections'] if x['type'] == 'SUBMESH'][0]
    vbindex = subvbs['data'][subindex]['vertexBufferIndex']
    ibindex = subvbs['data'][subindex]['indexBufferIndex']

    submesh = {}
    submesh['fmt'] = copy.deepcopy(fmts[vbindex])
    submesh['ib'] = generate_ib(ibindex, g1mg_stream, model_mesh_metadata, fmts, e=e)
    submesh['vb'] = generate_vb(vbindex, g1mg_stream, model_mesh_metadata, fmts, e=e)

    if submesh['ib'] is None:
        submesh['ib'] = []
    if submesh['vb'] is None:
        submesh['vb'] = []

    # Fix POSITION if needed
    stride_delta = 0
    for i, elem in enumerate(submesh['vb']):
        if elem['SemanticName'] == 'POSITION' and 'R16G16B16A16_UINT' in submesh['fmt']['elements'][i]['Format']:
            stride_delta = decode_nioh3_positions(elem, submesh['fmt']['elements'], bbox)
            submesh['fmt']['stride'] = str(int(submesh['fmt']['stride']) + stride_delta)
            for j in range(i + 1, len(submesh['fmt']['elements'])):
                submesh['fmt']['elements'][j]['AlignedByteOffset'] = str(
                    int(submesh['fmt']['elements'][j]['AlignedByteOffset']) + stride_delta
                )

    # Try meshlet path first (Nioh 3 GPU-driven meshes)
    meshlet_tris = _try_meshlet_path(subindex, g1mg_stream, model_mesh_metadata, fmts, e, bbox)
    if meshlet_tris is not None:
        submesh['ib'] = meshlet_tris
        submesh = cull_vb(submesh)
        return submesh

    # Fallback: for triangle list, generate_ib already returns correct triangles.
    # For triangle strip, read raw IB and split on large 3D jumps.
    topology = submesh['fmt'].get('topology', 'trianglelist')
    if topology == 'trianglestrip' and model_mesh_metadata.get('version') == 0x30303435:
        pos_idx = None
        for i, elem in enumerate(submesh['vb']):
            if elem.get('SemanticName') == 'POSITION':
                pos_idx = i
                break
        if pos_idx is not None:
            pos_buf = submesh['vb'][pos_idx]['Buffer']
            import math

            # Read raw IB indices as flat list
            ib_section = [x for x in model_mesh_metadata['sections'] if x['type'] == 'INDEX_BUFFER'][0]
            ib_entry = ib_section['data'][ibindex]
            with io.BytesIO(g1mg_stream) as f:
                f.seek(ib_entry['offset'])
                raw = f.read(ib_entry['stride'] * ib_entry['count'])
                if ib_entry['stride'] == 4:
                    flat = list(struct.unpack(e + 'I' * ib_entry['count'], raw))
                elif ib_entry['stride'] == 2:
                    flat = list(struct.unpack(e + 'H' * ib_entry['count'], raw))
                else:
                    flat = []

            if len(flat) < 3:
                submesh['ib'] = []
            else:
                def _dist(a, b):
                    return math.dist(pos_buf[a], pos_buf[b])
                dists = [_dist(flat[i], flat[i + 1]) for i in range(len(flat) - 1)]
                jump_thresh = 50.0
                jumps = [i for i, d in enumerate(dists) if d > jump_thresh]
                strips = []
                start = 0
                for j in jumps:
                    if j - start >= 2:
                        strips.append(flat[start:j + 1])
                    start = j + 1
                if len(flat) - start >= 3:
                    strips.append(flat[start:])
                filtered = []
                for strip in strips:
                    for k in range(len(strip) - 2):
                        if k % 2 == 0:
                            filtered.append([strip[k], strip[k + 1], strip[k + 2]])
                        else:
                            filtered.append([strip[k], strip[k + 2], strip[k + 1]])
                submesh['ib'] = filtered

    # Remove degenerate and duplicate triangles (fallback path may produce them)
    if submesh['ib']:
        cleaned = []
        seen = set()
        for tri in submesh['ib']:
            if tri[0] == tri[1] or tri[1] == tri[2] or tri[0] == tri[2]:
                continue
            key = tuple(sorted(tri))
            if key not in seen:
                seen.add(key)
                cleaned.append(tri)
        submesh['ib'] = cleaned

    return submesh


def dds_to_png(dds_path, png_path):
    """Convert a DDS file to PNG using texture2ddecoder if available."""
    try:
        import texture2ddecoder
    except ImportError:
        return False
    with open(dds_path, 'rb') as f:
        data = f.read()
    if data[:4] != b'DDS ':
        return False
    dwHeight = struct.unpack('<I', data[12:16])[0]
    dwWidth = struct.unpack('<I', data[16:20])[0]
    fourcc = data[84:88]
    if fourcc == b'DX10':
        dxgiFormat = struct.unpack('<I', data[128:132])[0]
        data_offset = 148
        if dxgiFormat == 71:      fmt = 'bc1'
        elif dxgiFormat == 77:    fmt = 'bc3'
        elif dxgiFormat == 80:    fmt = 'bc4'
        elif dxgiFormat == 83:    fmt = 'bc4'
        elif dxgiFormat == 95:    fmt = 'bc6'
        elif dxgiFormat == 98:    fmt = 'bc7'
        elif dxgiFormat == 2:     fmt = 'rgba32f'
        elif dxgiFormat == 41:    fmt = 'r32f'
        else:
            return False
    else:
        data_offset = 128
        if fourcc == b'DXT1':    fmt = 'bc1'
        elif fourcc == b'DXT5':  fmt = 'bc3'
        elif fourcc == b'ATI1':  fmt = 'bc4'
        elif fourcc == b'ATI2':  fmt = 'bc5'
        else:
            return False
    raw = data[data_offset:]
    if fmt == 'rgba32f':
        n = dwWidth * dwHeight
        floats = struct.unpack('<' + 'f' * (n * 4), raw[:n * 16])
        arr = numpy.zeros((dwHeight, dwWidth, 4), dtype=numpy.uint8)
        for i in range(n):
            y = i // dwWidth
            x = i % dwWidth
            arr[y, x] = [max(0, min(255, int(floats[i * 4 + j] * 255))) for j in range(4)]
        img = Image.fromarray(arr, 'RGBA')
    elif fmt == 'r32f':
        n = dwWidth * dwHeight
        floats = struct.unpack('<' + 'f' * n, raw[:n * 4])
        arr = numpy.zeros((dwHeight, dwWidth), dtype=numpy.uint8)
        for i in range(n):
            y = i // dwWidth
            x = i % dwWidth
            arr[y, x] = max(0, min(255, int(floats[i] * 255)))
        img = Image.fromarray(arr, 'L')
    else:
        decoder = getattr(texture2ddecoder, 'decode_' + fmt)
        decoded_bytes = decoder(raw, dwWidth, dwHeight)
        arr = numpy.frombuffer(decoded_bytes, dtype=numpy.uint8).reshape(dwHeight, dwWidth, 4)
        img = Image.fromarray(arr, 'RGBA')
    img.save(png_path)
    return True


def _get_image_info(path):
    """Return (width, height, pixel_value) for a 1x1 image, or (w, h, None) otherwise."""
    try:
        with Image.open(path) as img:
            w, h = img.size
            if w == 1 and h == 1:
                px = img.getpixel((0, 0))
                return w, h, px
            return w, h, None
    except Exception:
        return None, None, None


def export_textures_for_model(model_dir, out_dir):
    """
    Export all .g1t files in model_dir to DDS in out_dir, then convert to PNG.
    Also copies user-provided .png/.jpg files if present.
    Returns list of texture filenames ordered by global texture index.
    """
    g1t_files = sorted([f for f in os.listdir(model_dir) if f.lower().endswith('.g1t')])
    # Clean old texture files to avoid ID accumulation on re-runs
    for f in os.listdir(out_dir):
        if f.lower().endswith(('.dds', '.png', '.jpg', '.jpeg')):
            os.remove(os.path.join(out_dir, f))
    for g1t_file in g1t_files:
        g1t_path = os.path.join(model_dir, g1t_file)
        try:
            subprocess.run(
                [sys.executable, G1T_TOOL_PATH, 'export', g1t_path, out_dir],
                check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE
            )
        except subprocess.CalledProcessError as e:
            print("Warning: g1t_tool failed for {}: {}".format(g1t_file, e))
            continue
    # Collect and rename DDS files to sequential IDs
    dds_files = []
    existing_dds = sorted([f for f in os.listdir(out_dir) if f.lower().endswith('.dds')])
    for tex_idx, dds in enumerate(existing_dds):
        src = os.path.join(out_dir, dds)
        new_name = "{:03d}.dds".format(tex_idx)
        dst = os.path.join(out_dir, new_name)
        if src != dst:
            shutil.move(src, dst)
        dds_files.append(new_name)
    # Convert DDS to PNG for better glTF compatibility
    png_files = []
    for dds in dds_files:
        dds_path = os.path.join(out_dir, dds)
        png_name = os.path.splitext(dds)[0] + '.png'
        png_path = os.path.join(out_dir, png_name)
        if dds_to_png(dds_path, png_path):
            png_files.append(png_name)
        else:
            png_files.append(dds)
    # Copy user-provided image files (png/jpg) to output dir
    img_exts = ('.png', '.jpg', '.jpeg')
    user_images = sorted([f for f in os.listdir(model_dir) if f.lower().endswith(img_exts)])
    next_idx = len(png_files)
    for img_file in user_images:
        src = os.path.join(model_dir, img_file)
        ext = os.path.splitext(img_file)[1].lower()
        new_name = "{:03d}{}".format(next_idx, ext)
        dst = os.path.join(out_dir, new_name)
        shutil.copy2(src, dst)
        png_files.append(new_name)
        next_idx += 1
    return png_files


def generate_materials(gltf_data, model_mesh_metadata):
    """Generate placeholder materials without texture references.
    User will add images manually in the glTF viewer/editor."""
    metadata_sections = {model_mesh_metadata['sections'][i]['type']: i for i in range(len(model_mesh_metadata['sections']))}
    if 'MATERIALS' not in metadata_sections:
        return gltf_data
    materials = model_mesh_metadata['sections'][metadata_sections['MATERIALS']]['data']
    for i in range(len(materials)):
        material = {
            'name': 'Material_{0:02d}'.format(i),
            'doubleSided': True,
            'pbrMetallicRoughness': {
                'metallicFactor': 0.0,
                'roughnessFactor': 1.0
            }
        }
        # Foliage meshes typically rely on alpha test
        material['alphaMode'] = 'MASK'
        material['alphaCutoff'] = 0.5
        gltf_data['materials'].append(material)
    return gltf_data


def write_glTF(g1m_path, out_dir, g1mg_stream, model_mesh_metadata, model_skel_data, e='<'):
    g1m_name = os.path.splitext(os.path.basename(g1m_path))[0]
    gltf_path = os.path.join(out_dir, g1m_name + '.gltf')
    bin_path = os.path.join(out_dir, g1m_name + '.bin')

    metadata_sections = {model_mesh_metadata['sections'][i]['type']: i for i in range(len(model_mesh_metadata['sections']))}
    skel_present = model_skel_data['jointCount'] > 1 and not model_skel_data['boneList'][0]['parentID'] == -2147483648

    fmts = generate_fmts(model_mesh_metadata)
    gltf_data = {
        'asset': {'version': '2.0'},
        'accessors': [],
        'bufferViews': [],
        'buffers': [],
        'images': [],
        'materials': [],
        'meshes': [],
        'nodes': [],
        'samplers': [],
        'scenes': [{}],
        'scene': 0,
        'skins': [],
        'textures': []
    }
    gltf_data['scenes'][0]['nodes'] = [0]

    # Generate placeholder materials (no textures; user will add images manually)
    if 'MATERIALS' in metadata_sections:
        gltf_data = generate_materials(gltf_data, model_mesh_metadata)

    # Build skeleton nodes
    for i in range(len(model_skel_data['boneList'])):
        try:
            node = {'children': [], 'name': model_skel_data['boneList'][i]['bone_id']}
            if not list(model_skel_data['boneList'][i]['rotation_q']) == [0, 0, 0, 1]:
                node['rotation'] = model_skel_data['boneList'][i]['rotation_q']
            if not list(model_skel_data['boneList'][i]['scale']) == [1, 1, 1]:
                node['scale'] = model_skel_data['boneList'][i]['scale']
            tx, ty, tz = model_skel_data['boneList'][i]['pos_xyz']
            if tx != 0 or ty != 0 or tz != 0:
                node['translation'] = [tx, ty, tz]
            if i > 0:
                gltf_data['nodes'][model_skel_data['boneList'][i]['parentID']]['children'].append(len(gltf_data['nodes']))
            gltf_data['nodes'].append(node)
        except Exception:
            pass
    for i in range(len(gltf_data['nodes'])):
        if len(gltf_data['nodes'][i]['children']) == 0:
            del gltf_data['nodes'][i]['children']

    # Build meshes
    subvbs = model_mesh_metadata['sections'][metadata_sections['SUBMESH']]
    mesh_nodes = []
    giant_buffer = bytes()
    buffer_view = 0
    bbox = model_mesh_metadata['bounding_box']

    for subindex in range(len(subvbs['data'])):
        print("Processing submesh {}...".format(subindex))
        submesh = generate_nioh3_submesh(subindex, g1mg_stream, model_mesh_metadata, fmts, e=e, bbox=bbox)
        if len(submesh['ib']) == 0 or len(submesh['vb']) == 0:
            continue

        # Generate vgmap for skinning
        try:
            boneindex = subvbs['data'][subindex]['bonePaletteIndex']
            submesh['vgmap'] = generate_vgmap(boneindex, model_mesh_metadata, model_skel_data)
        except Exception:
            submesh['vgmap'] = {}

        skip_weights = True  # DEBUG: disable skinning for Nioh 3
        try:
            submesh = convert_bones_to_single_file(submesh)
            submesh = fix_weight_groups(submesh)
        except Exception:
            skip_weights = True

        if 'TANGENT' in [x['SemanticName'] for x in submesh['vb']]:
            submesh = fix_tangent_length(submesh)
        submesh = fix_normal_type(submesh)

        if translate_meshes and 'translation' in gltf_data['nodes'][0]:
            position_index = [x['SemanticName'] for x in submesh['fmt']['elements']].index('POSITION')
            position_veclength = len(submesh['vb'][position_index]['Buffer'][0])
            shift = numpy.array(gltf_data['nodes'][0]['translation'] + ([0] * (position_veclength - 3)))
            submesh['vb'][position_index]['Buffer'] = [(x + shift).tolist() for x in submesh['vb'][position_index]['Buffer']]

        gltf_fmt = convert_fmt_for_gltf(submesh['fmt'])
        vb_stream = io.BytesIO()
        write_vb_stream(submesh['vb'], vb_stream, gltf_fmt, e=e, interleave=False)
        block_offset = len(giant_buffer)

        primitive = {"attributes": {}}
        for element in range(len(gltf_fmt['elements'])):
            name = gltf_fmt['elements'][element]['SemanticName']
            if name[:5] in ['POSIT', 'WEIGH', 'JOINT', 'NORMA', 'COLOR', 'TEXCO', 'TANGE']:
                primitive["attributes"][name] = len(gltf_data['accessors'])
                gltf_data['accessors'].append({
                    "bufferView": buffer_view,
                    "componentType": gltf_fmt['elements'][element]['componentType'],
                    "count": len(submesh['vb'][element]['Buffer']),
                    "type": gltf_fmt['elements'][element]['accessor_type']
                })
                if name == 'POSITION':
                    gltf_data['accessors'][-1]['max'] = [
                        max([x[0] for x in submesh['vb'][element]['Buffer']]),
                        max([x[1] for x in submesh['vb'][element]['Buffer']]),
                        max([x[2] for x in submesh['vb'][element]['Buffer']])
                    ]
                    gltf_data['accessors'][-1]['min'] = [
                        min([x[0] for x in submesh['vb'][element]['Buffer']]),
                        min([x[1] for x in submesh['vb'][element]['Buffer']]),
                        min([x[2] for x in submesh['vb'][element]['Buffer']])
                    ]
                gltf_data['bufferViews'].append({
                    "buffer": 0,
                    "byteOffset": block_offset,
                    "byteLength": len(submesh['vb'][element]['Buffer']) * gltf_fmt['elements'][element]['componentStride'],
                    "target": 34962
                })
                block_offset += len(submesh['vb'][element]['Buffer']) * gltf_fmt['elements'][element]['componentStride']
                buffer_view += 1

        vb_stream.seek(0)
        giant_buffer += vb_stream.read()
        vb_stream.close()

        ib_stream = io.BytesIO()
        # Filter out incomplete triangles and flatten
        valid_tris = [tri for tri in submesh['ib'] if len(tri) == 3]
        flat_ib = [x for y in valid_tris for x in y]
        write_ib_stream(flat_ib, ib_stream, gltf_fmt, e=e)
        while (ib_stream.tell() % 4) > 0:
            ib_stream.write(b'\x00')
        primitive["indices"] = len(gltf_data['accessors'])
        gltf_data['accessors'].append({
            "bufferView": buffer_view,
            "componentType": gltf_fmt['componentType'],
            "count": len(flat_ib),
            "type": gltf_fmt['accessor_type']
        })
        gltf_data['bufferViews'].append({
            "buffer": 0,
            "byteOffset": len(giant_buffer),
            "byteLength": ib_stream.tell(),
            "target": 34963
        })
        buffer_view += 1
        ib_stream.seek(0)
        giant_buffer += ib_stream.read()
        ib_stream.close()

        primitive["mode"] = 4  # TRIANGLES
        mat_idx = subvbs['data'][subindex]['materialIndex']
        if mat_idx < len(gltf_data['materials']):
            primitive["material"] = mat_idx
            # Remove vertex colors (cloth transform semantic)
            primitive["attributes"] = {k: v for (k, v) in primitive["attributes"].items() if 'COLOR' not in k}

        mesh_nodes.append(len(gltf_data['nodes']))
        gltf_data['nodes'].append({'mesh': len(gltf_data['meshes']), 'name': "Mesh_{}".format(subindex)})
        gltf_data['meshes'].append({"primitives": [primitive], "name": "Mesh_{}".format(subindex)})

        if skel_present and not skip_weights:
            gltf_data['nodes'][-1]['skin'] = len(gltf_data['skins'])
            skin_bones = list_of_utilized_bones(submesh, model_skel_data)
            inv_mtx_buffer = bytes()
            for i in range(len(skin_bones)):
                mtx = Quaternion(model_skel_data['boneList'][skin_bones[i]]['abs_q']).transformation_matrix
                [mtx[0, 3], mtx[1, 3], mtx[2, 3]] = model_skel_data['boneList'][skin_bones[i]]['abs_p']
                inv_bind_mtx = numpy.linalg.inv(mtx)
                inv_bind_mtx = numpy.ndarray.transpose(inv_bind_mtx)
                inv_mtx_buffer += struct.pack(e + "16f", *[num for row in inv_bind_mtx for num in row])
            gltf_data['skins'].append({
                "inverseBindMatrices": len(gltf_data['accessors']),
                "joints": skin_bones
            })
            gltf_data['accessors'].append({
                "bufferView": buffer_view,
                "componentType": 5126,
                "count": len(skin_bones),
                "type": "MAT4"
            })
            gltf_data['bufferViews'].append({
                "buffer": 0,
                "byteOffset": len(giant_buffer),
                "byteLength": len(inv_mtx_buffer)
            })
            buffer_view += 1
            giant_buffer += inv_mtx_buffer

    gltf_data['scenes'][0]['nodes'].extend(mesh_nodes)
    gltf_data['buffers'].append({"byteLength": len(giant_buffer), "uri": g1m_name + '.bin'})

    with open(bin_path, 'wb') as f:
        f.write(giant_buffer)
    with open(gltf_path, 'w', encoding='utf-8') as f:
        json.dump(gltf_data, f, indent=4)
    print("Written:", gltf_path)


def process_g1m(g1m_path, out_dir):
    g1m_name = os.path.splitext(os.path.basename(g1m_path))[0]
    print("Processing {}...".format(g1m_path))
    with open(g1m_path, "rb") as f:
        file_magic, = struct.unpack(">I", f.read(4))
        if file_magic == 0x5F4D3147:
            e = '<'
        elif file_magic == 0x47314D5F:
            e = '>'
        else:
            print("Not a G1M file:", g1m_path)
            return False
        f.read(8)  # version, size
        chunks = {}
        chunks["starting_offset"], chunks["reserved"], chunks["count"] = struct.unpack(e + "III", f.read(12))
        f.seek(chunks["starting_offset"])
        g1mg_stream = None
        skel_stream = None
        for i in range(chunks["count"]):
            chunk_start = f.tell()
            magic = f.read(4).decode("utf-8")
            version = f.read(4).hex()
            size, = struct.unpack(e + "I", f.read(4))
            if magic in ['G1MG', 'GM1G']:
                f.seek(chunk_start)
                g1mg_stream = f.read(size)
            elif magic in ['G1MS', 'SM1G']:
                f.seek(chunk_start)
                skel_stream = f.read(size)
            else:
                f.seek(chunk_start + size)

    if g1mg_stream is None:
        print("No G1MG chunk found in", g1m_path)
        return False

    model_mesh_metadata = parseG1MG(g1mg_stream, e)

    if skel_stream:
        model_skel_data = parseG1MS(skel_stream, e)
        model_dir = os.path.dirname(g1m_path)
        oid_path = os.path.join(model_dir, g1m_name + 'Oid.bin')
        if os.path.exists(oid_path):
            model_skel_oid = binary_oid_to_dict(oid_path)
            model_skel_data = name_bones(model_skel_data, model_skel_oid)
        if model_skel_data['jointCount'] > 1 and not model_skel_data['boneList'][0]['parentID'] < -200000000:
            model_skel_data = calc_abs_skeleton(model_skel_data)
        else:
            # External skeleton not supported in this simplified converter
            pass
    else:
        model_skel_data = {'jointCount': 0, 'boneList': [], 'boneIDList': [], 'boneToBoneID': {}}

    os.makedirs(out_dir, exist_ok=True)
    write_glTF(g1m_path, out_dir, g1mg_stream, model_mesh_metadata, model_skel_data, e=e)
    return True


def main():
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Nioh 3 G1M to glTF converter')
        parser.add_argument('input', help='Input .g1m file or directory containing g1ms/ style models')
        parser.add_argument('output', nargs='?', default='gltf_out', help='Output directory (default: gltf_out)')
        args = parser.parse_args()
        input_path = args.input
        out_dir = args.output
    else:
        input_path = 'g1ms'
        out_dir = 'gltf_out'

    if os.path.isfile(input_path) and input_path.lower().endswith('.g1m'):
        process_g1m(input_path, out_dir)
    elif os.path.isdir(input_path):
        g1m_files = []
        for root, _dirs, files in os.walk(input_path):
            for f in files:
                if f.lower().endswith('.g1m'):
                    g1m_files.append(os.path.join(root, f))
        print("Found {} G1M files.".format(len(g1m_files)))
        for g1m_path in g1m_files:
            rel = os.path.relpath(os.path.dirname(g1m_path), input_path)
            target_dir = os.path.join(out_dir, rel)
            process_g1m(g1m_path, target_dir)
    else:
        print("Input must be a .g1m file or a directory.")
        sys.exit(1)


if __name__ == '__main__':
    main()
