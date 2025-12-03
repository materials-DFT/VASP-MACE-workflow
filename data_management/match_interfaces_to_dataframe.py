#!/usr/bin/env python3
"""
Match converged interfaces to specific DataFrame rows in pkl files.

This script includes a custom pickle loader that handles missing modules
and numpy version incompatibilities.
"""

import re
import sys
import pickle
import importlib
import pandas as pd
from pathlib import Path
from collections import defaultdict


class CustomUnpickler(pickle.Unpickler):
    """Custom unpickler that handles missing modules"""
    
    def find_class(self, module, name):
        # Handle numpy._core.numeric -> numpy.core.numeric
        if module == 'numpy._core.numeric':
            try:
                import numpy.core.numeric as numeric_module
                return getattr(numeric_module, name)
            except:
                pass
        
        # Handle missing slabgen module
        if module == 'slabgen':
            # Create a stub module
            if 'slabgen' not in sys.modules:
                stub_module = type(sys)('slabgen')
                sys.modules['slabgen'] = stub_module
            return getattr(sys.modules['slabgen'], name, None)
        
        # Try original import
        try:
            return super().find_class(module, name)
        except (ModuleNotFoundError, AttributeError) as e:
            # If module doesn't exist, create a stub
            if module not in sys.modules:
                stub_module = type(sys)(module)
                sys.modules[module] = stub_module
            
            # Try to get attribute, return None if not found
            mod = sys.modules[module]
            if hasattr(mod, name):
                return getattr(mod, name)
            else:
                # Create a stub class/function
                stub_class = type(name, (), {})
                setattr(mod, name, stub_class)
                return stub_class


def load_pkl_safe(pkl_file):
    """Load pickle file with fallback handling
    
    Tries multiple methods to load pickle files that may have compatibility issues:
    1. Standard pandas read_pickle
    2. Custom unpickler with stub handling
    3. Standard pickle with encoding
    4. Standard pickle
    
    Args:
        pkl_file: Path to pickle file
        
    Returns:
        Loaded DataFrame
        
    Raises:
        Exception: If all loading methods fail
    """
    pkl_path = Path(pkl_file).expanduser()
    
    # Try multiple methods
    methods = [
        # Method 1: Use pandas read_pickle
        lambda: pd.read_pickle(pkl_path),
        
        # Method 2: Custom unpickler
        lambda: CustomUnpickler(open(pkl_path, 'rb')).load(),
        
        # Method 3: Standard pickle with encoding
        lambda: pickle.load(open(pkl_path, 'rb'), encoding='latin1'),
        
        # Method 4: Standard pickle
        lambda: pickle.load(open(pkl_path, 'rb')),
    ]
    
    for i, method in enumerate(methods, 1):
        try:
            print(f"Trying method {i}...")
            data = method()
            print(f"Success with method {i}!")
            return data
        except Exception as e:
            print(f"Method {i} failed: {e}")
            continue
    
    raise Exception("All loading methods failed")

def extract_uuid_from_name(name):
    """Extract UUID from interface directory name"""
    match = re.search(r'([0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12})', name, re.IGNORECASE)
    return match.group(1).lower() if match else None

def parse_interface_name(name):
    """Parse interface name to extract components"""
    parts = name.split('.')
    if len(parts) < 2:
        return None
    
    name_part = parts[0]  # e.g., 'gamma_ramsdellite_101_111_0'
    uuid = parts[1] if len(parts) > 1 else None
    
    name_parts = name_part.split('_')
    if len(name_parts) < 5:
        return None
    
    phase1 = name_parts[0]
    phase2 = name_parts[1]
    # Parse miller indices - format is like "001_011_0" where:
    # - name_parts[2] = "001" -> hkl_A = (0,0,1)
    # - name_parts[3] = "011" -> hkl_B = (0,1,1)  
    # - name_parts[4] = "0" -> variant
    hkl_A_str = name_parts[2]  # e.g., "001"
    hkl_B_str = name_parts[3]  # e.g., "011"
    variant_str = name_parts[4] if len(name_parts) > 4 else "0"
    
    # Convert "001" -> (0,0,1)
    hkl_A = (int(hkl_A_str[0]), int(hkl_A_str[1]), int(hkl_A_str[2]))
    hkl_B = (int(hkl_B_str[0]), int(hkl_B_str[1]), int(hkl_B_str[2]))
    variant = int(variant_str)
    
    return {
        'phase1': phase1,
        'phase2': phase2,
        'phase_pair': f"{phase1}_{phase2}",
        'hkl_A': hkl_A,
        'hkl_B': hkl_B,
        'variant': variant,
        'uuid': uuid.lower() if uuid else None,
        'name': name
    }

def hkl_to_string(hkl):
    """Convert hkl tuple to string format"""
    if hkl is None:
        return None
    if isinstance(hkl, tuple) and len(hkl) == 3:
        return f"{hkl[0]}{hkl[1]}{hkl[2]}"
    return str(hkl)

def match_interface_to_dataframe(interface_info, df):
    """Match interface to DataFrame row"""
    matches = []
    
    # Strategy 1: Match by UUID in tag column
    if interface_info['uuid']:
        uuid = interface_info['uuid']
        if 'tag' in df.columns:
            tag_matches = df[df['tag'].astype(str).str.contains(uuid, case=False, na=False)]
            if len(tag_matches) > 0:
                for idx, row in tag_matches.iterrows():
                    matches.append({
                        'method': 'uuid_in_tag',
                        'index': idx,
                        'row': row
                    })
        
        # Also check if UUID is in index
        if df.index.dtype == 'object':
            idx_matches = df[df.index.astype(str).str.contains(uuid, case=False, na=False)]
            if len(idx_matches) > 0:
                for idx, row in idx_matches.iterrows():
                    matches.append({
                        'method': 'uuid_in_index',
                        'index': idx,
                        'row': row
                    })
    
    # Strategy 2: Match by hkl_A and phase pair
    hkl_A = interface_info['hkl_A']
    if hkl_A and 'hkl_A' in df.columns:
        hkl_matches = df[df['hkl_A'] == hkl_A]
        
        # If multiple matches, try to use variant to select the right one
        if len(hkl_matches) > 1 and interface_info['variant'] is not None:
            # Sort by index to get consistent ordering
            hkl_matches = hkl_matches.sort_index()
            if interface_info['variant'] < len(hkl_matches):
                hkl_matches = hkl_matches.iloc[[interface_info['variant']]]
        
        # If still multiple, take the first one
        if len(hkl_matches) > 0:
            for idx, row in hkl_matches.iterrows():
                matches.append({
                    'method': 'hkl_A',
                    'index': idx,
                    'row': row
                })
                break  # Only take first match
    
    # Strategy 3: Match by hkl_pair (both hkl_A and hkl_B)
    hkl_B = interface_info['hkl_B']
    if 'hkl_pair' in df.columns and hkl_A and hkl_B:
        # hkl_pair format: ((hkl_A), (hkl_B))
        def hkl_pair_matches(row):
            if pd.isna(row['hkl_pair']):
                return False
            hkl_pair = row['hkl_pair']
            if isinstance(hkl_pair, tuple) and len(hkl_pair) == 2:
                return hkl_pair[0] == hkl_A and hkl_pair[1] == hkl_B
            return False
        
        pair_matches = df[df.apply(hkl_pair_matches, axis=1)]
        # If multiple matches, use variant to select
        if len(pair_matches) > 1 and interface_info['variant'] is not None:
            pair_matches = pair_matches.sort_index()
            if interface_info['variant'] < len(pair_matches):
                pair_matches = pair_matches.iloc[[interface_info['variant']]]
        
        for idx, row in pair_matches.iterrows():
            matches.append({
                'method': 'hkl_pair',
                'index': idx,
                'row': row
            })
            break  # Only take first match
    
    return matches

def get_converged_interfaces(converged_dir):
    """Get list of converged interface directories"""
    interfaces = []
    for item in Path(converged_dir).iterdir():
        if item.is_dir():
            info = parse_interface_name(item.name)
            if info:
                info['path'] = str(item)
                interfaces.append(info)
    return interfaces

def main():
    pkl_base = Path.home() / "interfaces" / "interfaces" / "phase_phase"
    converged_dir = Path("/Users/924322630/palmetto_calculations/interfaces/neutralized/electronic_structure/converged")
    
    print("=" * 80)
    print("Loading converged interfaces...")
    print("=" * 80)
    converged = get_converged_interfaces(converged_dir)
    print(f"Total converged interfaces: {len(converged)}")
    
    print("\n" + "=" * 80)
    print("Matching to DataFrame rows...")
    print("=" * 80)
    
    all_matches = []
    no_match = []
    
    # Group by phase pair for efficiency
    by_phase_pair = defaultdict(list)
    for interface in converged:
        by_phase_pair[interface['phase_pair']].append(interface)
    
    for phase_pair, interfaces in sorted(by_phase_pair.items()):
        print(f"\nProcessing {phase_pair} ({len(interfaces)} interfaces)...")
        
        # Try both all_interfaces and selected_interfaces
        pkl_files = [
            pkl_base / phase_pair / "all_interfaces" / "interface_df.pkl",
            pkl_base / phase_pair / "selected_interfaces" / "interface_df.pkl"
        ]
        
        df = None
        pkl_source = None
        for pkl_file in pkl_files:
            if pkl_file.exists():
                try:
                    print(f"  Loading {pkl_file.name}...")
                    df = load_pkl_safe(str(pkl_file))
                    pkl_source = str(pkl_file)
                    print(f"  Loaded DataFrame with shape {df.shape}")
                    break
                except Exception as e:
                    print(f"  Failed to load {pkl_file}: {e}")
                    continue
        
        if df is None:
            print(f"  No pkl file found for {phase_pair}")
            no_match.extend(interfaces)
            continue
        
        # Check DataFrame structure
        print(f"  Columns: {list(df.columns)[:10]}...")
        if 'tag' in df.columns:
            print(f"  Sample tag values: {df['tag'].head(3).tolist()}")
        
        # Match each interface
        for interface in interfaces:
            matches = match_interface_to_dataframe(interface, df)
            
            if matches:
                # Use the first match (prefer UUID match if available)
                matches.sort(key=lambda x: 0 if 'uuid' in x['method'] else 1)
                match = matches[0]
                
                all_matches.append({
                    'converged_name': interface['name'],
                    'converged_path': interface['path'],
                    'uuid': interface['uuid'],
                    'phase_pair': phase_pair,
                    'hkl_A': interface['hkl_A'],
                    'pkl_source': pkl_source,
                    'match_method': match['method'],
                    'dataframe_index': match['index'],
                    'metadata': match['row'].to_dict()
                })
            else:
                no_match.append(interface)
    
    print("\n" + "=" * 80)
    print("Summary")
    print("=" * 80)
    print(f"Matched: {len(all_matches)}")
    print(f"Not matched: {len(no_match)}")
    
    # Print sample matches
    print("\n" + "=" * 80)
    print("Sample matches:")
    print("=" * 80)
    for match in all_matches[:5]:
        print(f"\nConverged: {match['converged_name']}")
        print(f"  UUID: {match['uuid']}")
        print(f"  Match method: {match['match_method']}")
        print(f"  DataFrame index: {match['dataframe_index']}")
        print(f"  PKL source: {match['pkl_source']}")
        print(f"  Key metadata:")
        metadata = match['metadata']
        for key in ['tag', 'hkl_A', 'hkl_B', 'hkl_pair', 'score', 'nat_interface']:
            if key in metadata:
                print(f"    {key}: {metadata[key]}")
    
    # Save results
    import json
    output_file = "interface_dataframe_matches.json"
    with open(output_file, 'w') as f:
        # Convert numpy types to native Python types for JSON serialization
        def convert_to_native(obj):
            import numpy as np
            # Skip complex objects that can't be serialized
            if hasattr(obj, '__class__') and obj.__class__.__name__ in ['Slab', 'Atoms']:
                return f"<{obj.__class__.__name__} object>"
            if isinstance(obj, (np.integer, np.floating)):
                return obj.item()
            elif isinstance(obj, np.ndarray):
                if obj.size == 1:
                    return obj.item()
                else:
                    return obj.tolist()
            elif hasattr(obj, 'item') and hasattr(obj, 'size') and obj.size == 1:
                return obj.item()
            elif hasattr(obj, 'tolist'):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_to_native(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_to_native(item) for item in obj]
            return obj
        
        json_data = []
        for match in all_matches:
            json_match = {
                'converged_name': match['converged_name'],
                'converged_path': match['converged_path'],
                'uuid': match['uuid'],
                'phase_pair': match['phase_pair'],
                'hkl_A': match['hkl_A'],
                'pkl_source': match['pkl_source'],
                'match_method': match['match_method'],
                'dataframe_index': int(match['dataframe_index']) if hasattr(match['dataframe_index'], 'item') else match['dataframe_index'],
                'metadata': convert_to_native(match['metadata'])
            }
            json_data.append(json_match)
        
        json.dump(json_data, f, indent=2)
    
    print(f"\nResults saved to {output_file}")
    
    # Also create CSV summary
    import pandas as pd
    csv_data = []
    for match in all_matches:
        csv_data.append({
            'converged_name': match['converged_name'],
            'uuid': match['uuid'],
            'phase_pair': match['phase_pair'],
            'hkl_A': str(match['hkl_A']),
            'match_method': match['match_method'],
            'dataframe_index': match['dataframe_index'],
            'pkl_source': match['pkl_source'],
            'tag': match['metadata'].get('tag', ''),
            'score': match['metadata'].get('score', ''),
            'nat_interface': match['metadata'].get('nat_interface', '')
        })
    
    df_summary = pd.DataFrame(csv_data)
    csv_file = "interface_dataframe_matches.csv"
    df_summary.to_csv(csv_file, index=False)
    print(f"CSV summary saved to {csv_file}")
    
    return all_matches, no_match

if __name__ == "__main__":
    # Allow standalone usage to load a single pkl file
    # Usage: python match_interfaces_to_dataframe.py <pkl_file>  (loads single file)
    #        python match_interfaces_to_dataframe.py             (runs matching)
    if len(sys.argv) > 1 and not sys.argv[1].startswith('--'):
        pkl_file = sys.argv[1]
        try:
            df = load_pkl_safe(pkl_file)
            print(f"\nLoaded DataFrame:")
            print(f"Shape: {df.shape}")
            print(f"Columns: {list(df.columns)}")
            print(f"\nFirst few rows:")
            print(df.head())
        except Exception as e:
            print(f"Failed to load: {e}")
            import traceback
            traceback.print_exc()
    else:
        # Run the main matching function
        all_matches, no_match = main()

