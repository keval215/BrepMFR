#!/usr/bin/env python3
"""
BrepMFR Exact Data Format Specification
This documents the EXACT format required by the trained model
"""

import torch
import dgl
import numpy as np

def create_exact_format_specification():
    """
    Documents the exact data format that BrepMFR expects
    Any conversion MUST match this format exactly
    """
    
    format_spec = {
        "bin_file_structure": {
            "description": "DGL graph saved with save_graphs()",
            "main_graph": "dgl.DGLGraph with node and edge data",
            "additional_data": "Dictionary with spatial relationship matrices"
        },
        
        "node_data_required": {
            "x": {
                "shape": "[num_nodes, U_grid, V_grid, point_features]",
                "dtype": "torch.float32",
                "description": "UV grid sampling of face geometry",
                "typical_size": "[N, 7, 7, 3]",  # Based on code analysis
                "content": "3D points sampled on face surface in UV coordinates"
            },
            "z": {
                "shape": "[num_nodes]", 
                "dtype": "torch.int32",
                "description": "Face surface type classification",
                "values": "0=other, 1=plane, 2=cylinder, 3=cone, 4=sphere, etc."
            },
            "y": {
                "shape": "[num_nodes]",
                "dtype": "torch.float32", 
                "description": "Face area in model units"
            },
            "l": {
                "shape": "[num_nodes]",
                "dtype": "torch.int32",
                "description": "Number of boundary loops per face"
            },
            "a": {
                "shape": "[num_nodes]", 
                "dtype": "torch.int32",
                "description": "Number of adjacent faces"
            },
            "f": {
                "shape": "[num_nodes]",
                "dtype": "torch.int32", 
                "description": "Ground truth feature labels (0-24)",
                "note": "This is what the model predicts!"
            }
        },
        
        "edge_data_required": {
            "x": {
                "shape": "[num_edges, U_grid, point_features]",
                "dtype": "torch.float32",
                "description": "UV sampling of edge geometry", 
                "typical_size": "[M, 7, 3]"
            },
            "t": {
                "shape": "[num_edges]",
                "dtype": "torch.int32",
                "description": "Edge type (line, arc, spline, etc.)"
            },
            "l": {
                "shape": "[num_edges]",
                "dtype": "torch.float32",
                "description": "Edge length"
            },
            "a": {
                "shape": "[num_edges]", 
                "dtype": "torch.float32",
                "description": "Dihedral angle between adjacent faces"
            },
            "c": {
                "shape": "[num_edges]",
                "dtype": "torch.int32", 
                "description": "Edge convexity (concave/convex/tangent)"
            }
        },
        
        "additional_matrices_required": {
            "edges_path": {
                "shape": "[num_nodes, num_nodes, max_dist]",
                "dtype": "torch.int32",
                "description": "Shortest path between faces through graph",
                "max_dist": 16,
                "fill_value": -1
            },
            "spatial_pos": {
                "shape": "[num_nodes, num_nodes]", 
                "dtype": "torch.int32",
                "description": "Spatial distance encoding between faces"
            },
            "d2_distance": {
                "shape": "[num_nodes, num_nodes, 64]",
                "dtype": "torch.float32", 
                "description": "2D distance features between faces"
            },
            "angle_distance": {
                "shape": "[num_nodes, num_nodes, 64]",
                "dtype": "torch.float32",
                "description": "Angular relationship features between faces"
            }
        },
        
        "json_file_structure": {
            "file_name": "String identifier matching bin file",
            "labels": "List[int] - ground truth labels for each face (0-24)"
        }
    }
    
    return format_spec

def validate_converted_file(bin_path, json_path):
    """
    Validate that a converted file matches the exact BrepMFR format
    """
    try:
        from dgl.data.utils import load_graphs
        import json
        
        # Load bin file
        graphs, graph_data = load_graphs(str(bin_path))
        graph = graphs[0]
        
        # Load json file  
        with open(json_path, 'r') as f:
            json_data = json.load(f)
        
        print(f"🔍 Validating: {bin_path}")
        
        # Check graph structure
        num_nodes = graph.num_nodes()
        num_edges = graph.num_edges()
        print(f"   Nodes: {num_nodes}, Edges: {num_edges}")
        
        # Validate node data
        required_node_keys = ['x', 'z', 'y', 'l', 'a', 'f']
        for key in required_node_keys:
            if key not in graph.ndata:
                print(f"   ❌ Missing node data: {key}")
                return False
            else:
                shape = graph.ndata[key].shape
                dtype = graph.ndata[key].dtype
                print(f"   ✅ node['{key}']: {shape}, {dtype}")
        
        # Validate edge data
        required_edge_keys = ['x', 't', 'l', 'a', 'c']
        for key in required_edge_keys:
            if key not in graph.edata:
                print(f"   ❌ Missing edge data: {key}")
                return False
            else:
                shape = graph.edata[key].shape
                dtype = graph.edata[key].dtype
                print(f"   ✅ edge['{key}']: {shape}, {dtype}")
        
        # Validate additional matrices
        required_matrices = ['edges_path', 'spatial_pos', 'd2_distance', 'angle_distance']
        for key in required_matrices:
            if key not in graph_data:
                print(f"   ❌ Missing matrix: {key}")
                return False
            else:
                shape = graph_data[key].shape
                dtype = graph_data[key].dtype
                print(f"   ✅ matrix['{key}']: {shape}, {dtype}")
        
        # Validate JSON structure
        if 'file_name' not in json_data:
            print(f"   ❌ Missing JSON field: file_name")
            return False
            
        if 'labels' not in json_data:
            print(f"   ❌ Missing JSON field: labels") 
            return False
        
        # Check label count matches node count
        if len(json_data['labels']) != num_nodes:
            print(f"   ❌ Label count mismatch: {len(json_data['labels'])} labels vs {num_nodes} nodes")
            return False
        
        print(f"   ✅ JSON: {len(json_data['labels'])} labels")
        print(f"✅ Validation PASSED!")
        return True
        
    except Exception as e:
        print(f"❌ Validation FAILED: {e}")
        return False

def compare_with_existing_file():
    """
    Compare your converted file with an existing working file
    """
    # Load a known working file for comparison
    working_file = "data/test/00000162.bin"
    working_json = "data/test/00000162.json"
    
    if not (Path(working_file).exists() and Path(working_json).exists()):
        print("❌ No working reference files found")
        return
    
    print("📋 Reference file structure:")
    validate_converted_file(working_file, working_json)

def create_format_checker():
    """
    Create a script to check if your conversion matches the expected format
    """
    script = '''
# Format Checker Usage:
# 1. Convert your STEP file to bin/json
# 2. Run: python format_checker.py your_file.bin your_file.json
# 3. Compare output with working reference files

if __name__ == "__main__":
    import sys
    if len(sys.argv) != 3:
        print("Usage: python format_checker.py <bin_file> <json_file>")
        sys.exit(1)
    
    bin_file = sys.argv[1] 
    json_file = sys.argv[2]
    
    if validate_converted_file(bin_file, json_file):
        print("🎉 Your file format is CORRECT!")
    else:
        print("❌ Your file format needs fixes")
'''
    
    return script

if __name__ == "__main__":
    print("📖 BrepMFR Exact Format Specification")
    print("=" * 50)
    
    spec = create_exact_format_specification()
    
    print("🔧 Critical Requirements:")
    print("1. UV grid sampling for faces AND edges")
    print("2. Geometric properties (area, length, angles)")
    print("3. Topological properties (face types, edge types)")
    print("4. Spatial relationship matrices")
    print("5. Ground truth labels matching face count")
    
    print("\n⚠️  The Challenge:")
    print("Creating UV grids and spatial matrices requires:")
    print("- Deep CAD geometry knowledge")
    print("- Complex geometric computations")
    print("- Exact format matching")
    
    print("\n💡 Recommendation:")
    print("1. Start with existing .bin file analysis")
    print("2. Reverse-engineer the exact format")
    print("3. Build conversion step-by-step")
    print("4. Validate against working files")
    
    # Show comparison with existing file
    compare_with_existing_file()
