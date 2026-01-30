"""
Filter NH3 LMDB to keep only MOFs with elements supported by the pretrained model.
"""
import lmdb
import pickle
from tqdm import tqdm

INPUT_LMDB = "raw_nh3_core/lmdbs/data.lmdb"
OUTPUT_LMDB = "raw_nh3_core/lmdbs/data_filtered.lmdb"
PROP_KEY = "NH3_uptake_298K_1bar [mmol/g]"

# Supported elements from pretrained BB encoder
SUPPORTED_ELEMENTS = {1, 2, 6, 7, 8, 9, 15, 16, 17, 23, 24, 28, 29, 30, 35, 53, 56}


def main():
    # Open input LMDB
    env_in = lmdb.open(INPUT_LMDB, readonly=True, lock=False, readahead=False, subdir=False)
    
    # Count and filter
    total = 0
    kept = 0
    kept_records = []
    
    with env_in.begin() as txn:
        for k, v in tqdm(txn.cursor(), desc="Filtering"):
            total += 1
            d = pickle.loads(v)
            
            # Check if has NH3 property
            pd = getattr(d, "prop_dict", None)
            if not isinstance(pd, dict) or PROP_KEY not in pd:
                continue
            
            # Check if has BBs
            if not hasattr(d, "bbs") or not d.bbs:
                continue
            
            # Check if all elements are supported
            all_supported = True
            for bb in d.bbs:
                elements = set(bb.atom_types.tolist())
                if not elements.issubset(SUPPORTED_ELEMENTS):
                    all_supported = False
                    break
            
            if all_supported:
                kept_records.append((k, v))
                kept += 1
    
    env_in.close()
    
    print(f"\nTotal records: {total}")
    print(f"Kept (with NH3 property and supported elements): {kept}")
    
    # Write filtered LMDB
    print(f"\nWriting filtered LMDB to {OUTPUT_LMDB}...")
    env_out = lmdb.open(OUTPUT_LMDB, map_size=int(1e11), subdir=False)
    
    with env_out.begin(write=True) as txn:
        for i, (k, v) in enumerate(tqdm(kept_records, desc="Writing")):
            # Use sequential integer keys
            txn.put(f"{i}".encode("ascii"), v)
        # Store length
        txn.put("length".encode("ascii"), pickle.dumps(len(kept_records)))
    
    env_out.close()
    print(f"Done! Saved {kept} records to {OUTPUT_LMDB}")


if __name__ == "__main__":
    main()
