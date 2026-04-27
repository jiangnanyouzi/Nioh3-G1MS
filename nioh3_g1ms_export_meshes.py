#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Nioh 3 G1M Mesh Exporter (fmt/ib/vb/vgmap)

Parses Nioh 3 G1M files (G1MG v0x30303435) and exports submeshes as
.fmt + .ib + .vb (+ .vgmap) files compatible with lib_fmtibvb and Blender.

Handles Nioh 3 specific quirks:
  - POSITION stored as R16G16B16A16_UINT (decoded as UNORM mapped to bounding box)
  - SUBMESH metadata vertex/index counts are unreliable; uses full VB/IB buffers
  - Meshlet sections (0x1000b/0x1000c/0x1000e) for GPU-driven geometry

Usage:
    python nioh3_g1ms_export_meshes.py <g1m_file_or_dir> [options]

Options:
    -o, --overwrite          Overwrite existing output folder
    -f, --full_vertices      Output full meshes (do NOT cull unreferenced vertices)

GitHub jiangnanyouzi/Nioh3-G1MS
"""

import glob
import os
import io
import sys
import re
import copy
import json
import struct
import math

_project_root = os.path.dirname(os.path.abspath(__file__))
if _project_root not in sys.path:
    sys.path.insert(0, _project_root)

try:
    import numpy
    from lib_fmtibvb import write_fmt, write_ib, write_vb, get_stride_from_dxgi_format
    from g1m_export_meshes import (
        parseG1MG, parseG1MS, generate_fmts, generate_ib, generate_vb,
        calc_abs_skeleton, combine_skeleton, binary_oid_to_dict, name_bones,
        generate_vgmap, cull_vb, get_ext_skeleton
    )
except ModuleNotFoundError as e:
    print("Python module missing! {}".format(e.msg))
    sys.exit(1)


# ---------------------------------------------------------------------------
# Nioh 3 specific helpers (mirrored from nioh3_g1m_to_gltf.py)
# ---------------------------------------------------------------------------

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
        x = bbox['min_x'] + (raw[0] / 65535.0) * (bbox['max_x'] - bbox['min_x'])
        y = bbox['min_y'] + (raw[1] / 65535.0) * (bbox['max_y'] - bbox['min_y'])
        z = bbox['min_z'] + (raw[2] / 65535.0) * (bbox['max_z'] - bbox['min_z'])
        new_buffer.append([x, y, z])
    vb_element['Buffer'] = new_buffer
    vb_element['Format'] = 'R32G32B32_FLOAT'
    for elem in fmt_elements:
        if elem['SemanticName'] == 'POSITION':
            elem['Format'] = 'R32G32B32_FLOAT'
    return new_stride - old_stride


def _parse_g1mg_section(g1mg_stream, target_magic, e='<'):
    """Parse a single section from G1MG stream by magic."""
    with io.BytesIO(g1mg_stream) as f:
        f.read(4); f.read(8); f.read(4); f.read(28)
        section_count = struct.unpack(e + 'I', f.read(4))[0]
        for _i in range(section_count):
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
            if e_count == c_count and e_stride >= 24:
                leaf_meshlets = set()
                for i in range(e_count):
                    off = 8 + i * e_stride
                    child0, child1 = struct.unpack(e + 'ii', e_data[off + 16:off + 24])
                    if child0 == -1 and child1 == -1:
                        leaf_meshlets.add(i)

    # Parse 0x10010 for meshlet-to-submesh mapping
    sec_10 = _parse_g1mg_section(g1mg_stream, 0x00010010, e)
    meshlet_to_submesh = None
    if sec_10:
        data_10 = sec_10['data']
        if len(data_10) >= 8:
            count_10, stride_10 = struct.unpack(e + 'II', data_10[:8])
            if count_10 == c_count and stride_10 >= 12:
                meshlet_to_submesh = {}
                for i in range(count_10):
                    off = 8 + i * stride_10
                    a, b, c_val = struct.unpack(e + 'III', data_10[off:off + 12])
                    submesh_id = (c_val >> 16) & 0xFFFF
                    meshlet_to_submesh[i] = submesh_id

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
    If meshlet sections are present, uses them; otherwise falls back to strip parsing.
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

    # Fix POSITION if quantized
    stride_delta = 0
    for i, elem in enumerate(submesh['vb']):
        if elem['SemanticName'] == 'POSITION' and 'R16G16B16A16_UINT' in submesh['fmt']['elements'][i]['Format']:
            stride_delta = decode_nioh3_positions(elem, submesh['fmt']['elements'], bbox)
            submesh['fmt']['stride'] = str(int(submesh['fmt']['stride']) + stride_delta)
            for j in range(i + 1, len(submesh['fmt']['elements'])):
                submesh['fmt']['elements'][j]['AlignedByteOffset'] = str(
                    int(submesh['fmt']['elements'][j]['AlignedByteOffset']) + stride_delta
                )

    # Try meshlet path first
    meshlet_tris = _try_meshlet_path(subindex, g1mg_stream, model_mesh_metadata, fmts, e, bbox)
    if meshlet_tris is not None:
        submesh['ib'] = meshlet_tris
        submesh = cull_vb(submesh)
        return submesh

    # Fallback strip parsing
    topology = submesh['fmt'].get('topology', 'trianglelist')
    if topology == 'trianglestrip' and model_mesh_metadata.get('version') == 0x30303435:
        pos_idx = None
        for i, elem in enumerate(submesh['vb']):
            if elem.get('SemanticName') == 'POSITION':
                pos_idx = i
                break
        if pos_idx is not None:
            pos_buf = submesh['vb'][pos_idx]['Buffer']
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

            if len(flat) >= 3:
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

    # Remove degenerate and duplicate triangles
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


# ---------------------------------------------------------------------------
# Write exported submeshes (fmt/ib/vb/vgmap)
# ---------------------------------------------------------------------------

def _sanitize_blend_for_blender(submesh):
    """
    blender_3dmigoto.py crashes if BLENDWEIGHT exists without BLENDINDICES
    (or vice versa). Remove orphaned blend semantics.
    Also removes blend data for meshes with no skeleton.
    """
    fmt = submesh['fmt']
    vb = submesh['vb']
    has_bi = any(e['SemanticName'] == 'BLENDINDICES' for e in fmt['elements'])
    has_bw = any(e['SemanticName'] == 'BLENDWEIGHT' for e in fmt['elements'])

    if not (has_bi and has_bw):
        # Remove any orphaned blend elements
        indices_to_remove = []
        for i, e in enumerate(fmt['elements']):
            if e['SemanticName'] in ('BLENDINDICES', 'BLENDWEIGHT'):
                indices_to_remove.append(i)
        if indices_to_remove:
            # Remove from vb (same indices as fmt elements)
            for i in sorted(indices_to_remove, reverse=True):
                del vb[i]
            # Remove from fmt and recompute offsets / stride
            new_elements = []
            offset = 0
            for i, e in enumerate(fmt['elements']):
                if i not in indices_to_remove:
                    new_e = copy.deepcopy(e)
                    new_e['id'] = str(len(new_elements))
                    new_e['AlignedByteOffset'] = str(offset)
                    stride_add = get_stride_from_dxgi_format('DXGI_FORMAT_' + new_e['Format'])
                    if stride_add:
                        offset += stride_add
                    new_elements.append(new_e)
            fmt['elements'] = new_elements
            fmt['stride'] = str(offset)
    return submesh


def write_nioh3_submeshes(g1mg_stream, model_mesh_metadata, model_skel_data, path='',
                          e='<', cull_vertices=True):
    """Write all submeshes as .fmt / .ib / .vb / .vgmap files."""
    subvbs = [x for x in model_mesh_metadata['sections'] if x['type'] == 'SUBMESH'][0]
    fmts = generate_fmts(model_mesh_metadata)
    bbox = model_mesh_metadata['bounding_box']

    for subindex in range(len(subvbs['data'])):
        print("Processing submesh {}...".format(subindex))
        submesh = generate_nioh3_submesh(subindex, g1mg_stream, model_mesh_metadata,
                                         fmts, e=e, bbox=bbox)

        # Sanitize for blender_3dmigoto compatibility
        submesh = _sanitize_blend_for_blender(submesh)

        write_fmt(submesh['fmt'], '{0}{1}.fmt'.format(path, subindex))
        if len(submesh['ib']) > 0:
            write_ib(submesh['ib'], '{0}{1}.ib'.format(path, subindex), submesh['fmt'])
            write_vb(submesh['vb'], '{0}{1}.vb'.format(path, subindex), submesh['fmt'])

        # vgmap
        if model_skel_data['jointCount'] > 1 and not model_skel_data['boneList'][0]['parentID'] < -200000000:
            try:
                boneindex = subvbs['data'][subindex]['bonePaletteIndex']
                vgmap = generate_vgmap(boneindex, model_mesh_metadata, model_skel_data)
                with open('{0}{1}.vgmap'.format(path, subindex), 'wb') as f:
                    f.write(json.dumps(vgmap, indent=4).encode("utf-8"))
            except Exception:
                pass

        # mesh_metadata.json (one per model, not per submesh)
        if subindex == 0:
            with open('{0}mesh_metadata.json'.format(path), 'wb') as f:
                f.write(json.dumps(model_mesh_metadata, indent=4).encode("utf-8"))


# ---------------------------------------------------------------------------
# Main parser (similar to g1m_export_meshes.parseG1M)
# ---------------------------------------------------------------------------

def parseNioh3G1M(g1m_name, overwrite=False, cull_vertices=True):
    """
    g1m_name: path to .g1m file WITHOUT extension, OR full path with extension.
    """
    if g1m_name.lower().endswith('.g1m'):
        g1m_path = g1m_name
        g1m_name = os.path.splitext(os.path.basename(g1m_path))[0]
    else:
        g1m_path = g1m_name + '.g1m'

    if not os.path.exists(g1m_path):
        print("File not found:", g1m_path)
        return False

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
        for _i in range(chunks["count"]):
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
            ext_skel = get_ext_skeleton(g1m_name)
            if ext_skel is not False:
                model_skel_data = combine_skeleton(ext_skel, model_skel_data)
    else:
        model_skel_data = {'jointCount': 0, 'boneList': [], 'boneIDList': [], 'boneToBoneID': {}}

    out_dir = g1m_name
    if os.path.exists(out_dir) and os.path.isdir(out_dir) and not overwrite:
        resp = input(out_dir + " folder exists! Overwrite? (y/N) ")
        if resp.strip().lower().startswith('y'):
            overwrite = True

    if overwrite or not os.path.exists(out_dir):
        if not os.path.exists(out_dir):
            os.mkdir(out_dir)
        write_nioh3_submeshes(g1mg_stream, model_mesh_metadata, model_skel_data,
                              path=out_dir + '/', e=e, cull_vertices=cull_vertices)
        print("Done. Output in:", os.path.abspath(out_dir))
    return True


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

def main():
    if len(sys.argv) > 1:
        import argparse
        parser = argparse.ArgumentParser(description='Nioh 3 G1M mesh exporter (fmt/ib/vb)')
        parser.add_argument('-o', '--overwrite', help="Overwrite existing files", action="store_true")
        parser.add_argument('-f', '--full_vertices',
                            help="Output full meshes instead of submeshes (do not cull unreferenced vertices)",
                            action="store_false")
        parser.add_argument('g1m_filename', help="Name of g1m file to extract meshes / G1MG metadata (required).")
        args = parser.parse_args()

        if os.path.exists(args.g1m_filename) and args.g1m_filename[-4:].lower() == '.g1m':
            parseNioh3G1M(args.g1m_filename[:-4], overwrite=args.overwrite, cull_vertices=args.full_vertices)
        else:
            print("File not found or not a .g1m:", args.g1m_filename)
            sys.exit(1)
    else:
        # Batch mode: scan g1m2/ directory
        g1m_files = []
        for root, _dirs, files in os.walk('g1ms'):
            for f in files:
                if f.lower().endswith('.g1m'):
                    g1m_files.append(os.path.join(root, f))
        if g1m_files:
            print("Found {} G1M files in g1ms/".format(len(g1m_files)))
            for g1m_path in g1m_files:
                g1m_name = os.path.splitext(g1m_path)[0]
                parseNioh3G1M(g1m_name, overwrite=False)
        else:
            g1m_files = glob.glob('*.g1m')
            if len(g1m_files) == 1:
                parseNioh3G1M(g1m_files[0][:-4])
            elif len(g1m_files) > 1:
                print('Which g1m file do you want to unpack?\n')
                for i, g in enumerate(g1m_files):
                    print(str(i + 1) + '. ' + g)
                choice = -1
                while choice < 0 or choice >= len(g1m_files):
                    try:
                        choice = int(input("\nPlease enter which g1m file to use:  ")) - 1
                    except ValueError:
                        pass
                if choice in range(len(g1m_files)):
                    parseNioh3G1M(g1m_files[choice][:-4])
            else:
                print("No .g1m files found.")


if __name__ == '__main__':
    main()
