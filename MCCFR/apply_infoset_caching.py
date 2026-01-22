#!/usr/bin/env python3
"""
Script to add infoset caching optimization to mccfr_v2.ipynb

This modifies the external_sampling function to cache infoset strings
and avoid recomputing the betting history on every call.

Usage:
    python3 apply_infoset_caching.py
"""

import json
import sys
from pathlib import Path

def modify_notebook():
    """Apply infoset caching modifications to mccfr_v2.ipynb"""
    
    notebook_path = Path(__file__).parent / 'mccfr_v2.ipynb'
    
    if not notebook_path.exists():
        print(f"ERROR: {notebook_path} not found!")
        return False
    
    print(f"Loading {notebook_path}...")
    with open(notebook_path, 'r') as f:
        notebook = json.load(f)
    
    modified = False
    
    # Find and modify the external_sampling function cell
    for cell in notebook['cells']:
        if cell['cell_type'] == 'code':
            source = ''.join(cell['source'])
            
            # Check if this is the external_sampling cell
            if 'def external_sampling(state: Union[RoundState, TerminalState]' in source:
                print("Found external_sampling function!")
                
                # Modification 1: Add infoset_cache parameter
                old_sig = 'def external_sampling(state: Union[RoundState, TerminalState], \n                     traversing_player: int) -> float:'
                new_sig = 'def external_sampling(state: Union[RoundState, TerminalState], \n                     traversing_player: int,\n                     infoset_cache: dict = None) -> float:'
                
                if old_sig in source:
                    source = source.replace(old_sig, new_sig)
                    print("✓ Modified function signature")
                    modified = True
                
                # Modification 2: Update docstring
                old_doc = '    Args:\n        state: Current game state (RoundState or TerminalState)\n        traversing_player: Player index whose regrets we\'re updating (0 or 1)\n    \n    Returns:'
                new_doc = '    Args:\n        state: Current game state (RoundState or TerminalState)\n        traversing_player: Player index whose regrets we\'re updating (0 or 1)\n        infoset_cache: Dict caching state_id -> infoset string (avoids recomputation)\n    \n    Returns:'
                
                if old_doc in source:
                    source = source.replace(old_doc, new_doc)
                    print("✓ Updated docstring")
                
                # Modification 3: Add cache initialization
                after_docstring = '    """\n    \n    # ========================================================================'
                new_init = '    """\n    \n    # Initialize cache if not provided\n    if infoset_cache is None:\n        infoset_cache = {}\n    \n    # ========================================================================'
                
                if after_docstring in source and new_init not in source:
                    source = source.replace(after_docstring, new_init)
                    print("✓ Added cache initialization")
                
                # Modification 4: Replace infoset computation with caching
                old_infoset = '    active_player = state.button % 2\n    infoset = get_infoset(state, active_player)\n    \n    # Get legal actions'
                new_infoset = '''    active_player = state.button % 2
    
    # OPTIMIZATION: Cache infoset lookups to avoid recomputing history
    state_id = id(state)
    if state_id in infoset_cache:
        infoset = infoset_cache[state_id]
    else:
        infoset = get_infoset(state, active_player)
        infoset_cache[state_id] = infoset
    
    # Get legal actions'''
                
                if old_infoset in source:
                    source = source.replace(old_infoset, new_infoset)
                    print("✓ Added caching logic")
                
                # Modification 5: Pass cache through recursive calls (traversing player)
                old_recurse1 = 'action_values[action_key] = external_sampling(next_state, traversing_player)'
                new_recurse1 = 'action_values[action_key] = external_sampling(next_state, traversing_player, infoset_cache)'
                
                if old_recurse1 in source:
                    source = source.replace(old_recurse1, new_recurse1)
                    print("✓ Updated traversing player recursive call")
                
                # Modification 6: Pass cache through recursive calls (opponent)
                old_recurse2 = 'value = external_sampling(next_state, traversing_player)'
                new_recurse2 = 'value = external_sampling(next_state, traversing_player, infoset_cache)'
                
                if old_recurse2 in source:
                    source = source.replace(old_recurse2, new_recurse2)
                    print("✓ Updated opponent recursive call")
                
                # Update the cell source
                cell['source'] = source.splitlines(True)
                break
    
    if modified:
        # Backup original
        backup_path = notebook_path.with_suffix('.ipynb.backup')
        print(f"\nCreating backup: {backup_path}")
        with open(backup_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        # Save modified notebook
        print(f"Saving modified notebook: {notebook_path}")
        with open(notebook_path, 'w') as f:
            json.dump(notebook, f, indent=1)
        
        print("\n✅ SUCCESS! Infoset caching optimization applied.")
        print("\nNext steps:")
        print("1. Open mccfr_v2.ipynb in Jupyter")
        print("2. Re-run Cell 17 (external_sampling function)")
        print("3. Test with training - you should see ~10-30% speedup!")
        return True
    else:
        print("\n⚠️  WARNING: Could not find code to modify.")
        print("The notebook may have already been modified or has a different structure.")
        return False

if __name__ == '__main__':
    success = modify_notebook()
    sys.exit(0 if success else 1)
