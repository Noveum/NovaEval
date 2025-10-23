#!/usr/bin/env python3
"""
Script to fix api_data.json by adding item_key field from turn_id found anywhere in the item.
"""

import json

def fix_api_data():
    """Fix api_data.json by adding item_key field"""
    
    # Load the api_data.json file
    with open('api_data.json', 'r') as f:
        data = json.load(f)
    
    # Process each item to add item_key
    items = data.get('items', [])
    fixed_items = []
    
    for item in items:
        # Create a copy of the item
        fixed_item = item.copy()
        
        # Look for turn_id in any field
        turn_id = None
        
        # Check if turn_id is directly in the item
        if 'turn_id' in item:
            turn_id = item['turn_id']
        else:
            # Search through all string values for turn_id pattern
            for key, value in item.items():
                if isinstance(value, str) and 'turn_id' in value:
                    try:
                        # Try to parse as JSON and extract turn_id
                        parsed = json.loads(value)
                        if isinstance(parsed, dict) and 'turn_id' in parsed:
                            turn_id = parsed['turn_id']
                            break
                    except:
                        pass
        
        # Add item_key field
        fixed_item['item_key'] = turn_id or ''
        fixed_items.append(fixed_item)
    
    # Update the data
    data['items'] = fixed_items
    
    # Save the fixed file
    with open('api_data.json', 'w') as f:
        json.dump(data, f, indent=2)
    
    print(f"Fixed api_data.json with {len(fixed_items)} items")
    print(f"Added item_key field for each item using turn_id found in the data")
    
    # Show a sample of the fixed data
    if fixed_items:
        print(f"\nSample item_key: {fixed_items[0].get('item_key', 'NOT_FOUND')}")

if __name__ == "__main__":
    fix_api_data()
