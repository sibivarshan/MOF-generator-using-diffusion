import lmdb, pickle
import copy
import torch
from tqdm import tqdm
from torch_geometric.loader import DataLoader
from torch_geometric.data import Data, Batch
from mofdiff.common.eval_utils import load_mofdiff_model
from mofdiff.common.data_utils import frac_to_cart_coords

PROP_KEY = "NH3_uptake_298K_1bar [mmol/g]"
LMDB_PATH = "raw_nh3_core/lmdbs/data.lmdb"

# CHANGE THIS to your pretrained model dir (the one you used earlier)
MODEL_PATH = "/home/sibivarshan_m7/MOFDiff/MOFDiff/pretrained/mofdiff_ckpt"


OUT_PATH = "raw_nh3_core/nh3_latent_dataset.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BB_EMB_CLIPPING = 100.0
BB_BATCH_SIZE = 32  # Reduced batch size
MAX_ATOMS = 1000
MAX_CPS = 20

# Supported elements from the pretrained BB encoder type mapper
SUPPORTED_ELEMENTS = {1, 2, 6, 7, 8, 9, 15, 16, 17, 23, 24, 28, 29, 30, 35, 53, 56}


def check_elements_supported(bbs):
    """Check if all elements in BBs are supported by the type mapper"""
    for bb in bbs:
        elements = set(bb.atom_types.tolist())
        if not elements.issubset(SUPPORTED_ELEMENTS):
            return False
    return True


def preprocess_bb(bb):
    """Preprocess a building block, mimicking MOFDataset.bb_criterion
    
    Returns a new BB object to avoid modifying the original.
    """
    # Deep copy the BB to avoid modifying the original
    bb = copy.deepcopy(bb)
    
    # Ensure all tensors are on CPU for preprocessing
    bb = bb.cpu() if hasattr(bb, 'cpu') else bb
    
    # Ensure num_atoms is set
    if not hasattr(bb, "num_atoms") or bb.num_atoms is None:
        bb.num_atoms = bb.atom_types.shape[0]
    
    # Ensure num_atoms is a scalar tensor
    if isinstance(bb.num_atoms, int):
        bb.num_atoms = torch.tensor(bb.num_atoms, dtype=torch.long)
    elif hasattr(bb.num_atoms, 'item'):
        bb.num_atoms = torch.tensor(bb.num_atoms.item(), dtype=torch.long)
    
    # Set num_nodes
    bb.num_nodes = bb.num_atoms.item() if hasattr(bb.num_atoms, 'item') else bb.num_atoms
    
    # Calculate num_cps (number of connection points)
    if hasattr(bb, 'is_anchor'):
        bb.num_cps = bb.is_anchor.long().sum()
        # Ensure it's a scalar tensor
        if hasattr(bb.num_cps, 'item'):
            bb.num_cps = torch.tensor(bb.num_cps.item(), dtype=torch.long)
    else:
        bb.num_cps = torch.tensor(0, dtype=torch.long)
    
    # Check size limits
    num_atoms_val = bb.num_atoms.item() if hasattr(bb.num_atoms, 'item') else bb.num_atoms
    num_cps_val = bb.num_cps.item() if hasattr(bb.num_cps, 'item') else bb.num_cps
    if num_atoms_val > MAX_ATOMS or num_cps_val > MAX_CPS:
        return None
    
    # Calculate cartesian coordinates and diameter
    try:
        cart_coords = frac_to_cart_coords(
            bb.frac_coords, bb.lengths, bb.angles, bb.num_atoms.unsqueeze(0)
        )
        pdist = torch.cdist(cart_coords.unsqueeze(0), cart_coords.unsqueeze(0)).squeeze()
        if pdist.numel() > 1:
            bb.diameter = pdist[pdist > 0].max() if (pdist > 0).any() else torch.tensor(0.0)
        else:
            bb.diameter = torch.tensor(0.0)
    except Exception as e:
        print(f"Error computing diameter: {e}")
        bb.diameter = torch.tensor(0.0)
    
    return bb


def embed_bbs(bb_encoder, bbs, device):
    """Embed building blocks using the BB encoder, mimicking MOFDataset.embed_bb"""
    n_batches = (len(bbs) + BB_BATCH_SIZE - 1) // BB_BATCH_SIZE
    all_bb_emb = []
    
    for i in range(n_batches):
        batch_bbs = bbs[i * BB_BATCH_SIZE : (i + 1) * BB_BATCH_SIZE]
        try:
            # Create batch on CPU first
            batch = Batch.from_data_list(batch_bbs)
            # Debug print
            print(f"  BB batch {i}: n_bbs={len(batch_bbs)}, lengths shape={batch.lengths.shape}, num_atoms shape={batch.num_atoms.shape}, frac_coords shape={batch.frac_coords.shape}")
            # Now move to device
            batch = batch.to(device)
            batch.atom_types = bb_encoder.type_mapper.transform(batch.atom_types)
            with torch.no_grad():
                bb_emb = bb_encoder.encode(batch)
            all_bb_emb.append(bb_emb.cpu())
            
            # Synchronize CUDA after each batch
            if torch.cuda.is_available():
                torch.cuda.synchronize()
                
        except Exception as e:
            print(f"Error in BB batch {i}/{n_batches}, batch size={len(batch_bbs)}")
            print(f"  Error: {e}")
            print(f"  Batched lengths shape: {batch.lengths.shape}")
            print(f"  Batched angles shape: {batch.angles.shape}")
            print(f"  Batched num_atoms: {batch.num_atoms}")
            print(f"  Batched frac_coords shape: {batch.frac_coords.shape}")
            # Print debug info about the BBs
            for j, bb in enumerate(batch_bbs[:5]):  # First 5
                print(f"  BB {j}: num_atoms={bb.num_atoms}, lengths shape={bb.lengths.shape}, atom_types unique={torch.unique(bb.atom_types).tolist()}")
            raise
    
    all_bb_emb = torch.cat(all_bb_emb, dim=0)
    all_bb_emb = torch.clamp(all_bb_emb, -BB_EMB_CLIPPING, BB_EMB_CLIPPING)
    return all_bb_emb


def prepare_cg_data(raw_data, bb_emb):
    """Convert raw MOF data to CG format expected by the encoder, mimicking MOFDataset.get_cg"""
    cell = raw_data.cell
    if cell.dim() == 2:
        cell = cell.unsqueeze(0)
    
    cg_data = Data(
        frac_coords=raw_data.cg_frac_coords,
        lengths=raw_data.lengths,
        angles=raw_data.angles,
        cell=cell,
        edge_index=raw_data.cg_edge_index,
        to_jimages=raw_data.cg_to_jimages,
        num_bonds=raw_data.num_cg_bonds,
        num_atoms=raw_data.num_components,
        num_nodes=raw_data.num_components,
        scaled_lattice=raw_data.scaled_lattice,
        num_components=raw_data.num_components,
        is_linker=raw_data.is_linker,
        n_metal=raw_data.is_linker.shape[0] - raw_data.is_linker.sum(),
        n_linker=raw_data.is_linker.sum(),
        node_embedding=True,
        atom_types=bb_emb,  # This is the BB embedding!
    )
    return cg_data


def main():
    model, cfg, bb_encoder = load_mofdiff_model(MODEL_PATH)
    model = model.to(DEVICE).eval()
    bb_encoder = bb_encoder.to(DEVICE).eval()
    bb_encoder.type_mapper.match_device(bb_encoder)

    env = lmdb.open(LMDB_PATH, readonly=True, lock=False, readahead=False, subdir=False)

    z_chunks = []
    y_list = []
    id_list = []

    batch = []
    batch_y = []
    batch_id = []
    batch_bbs = []  # Track BBs for each sample
    batch_n_bbs = []  # Track number of BBs per sample

    def flush():
        nonlocal batch, batch_y, batch_id, batch_bbs, batch_n_bbs
        if not batch:
            return
        
        print(f"\nFlush: processing {len(batch)} MOFs")
        
        # First, embed all BBs in this batch
        all_bbs_flat = [bb for bbs in batch_bbs for bb in bbs]
        if not all_bbs_flat:
            batch, batch_y, batch_id, batch_bbs, batch_n_bbs = [], [], [], [], []
            return
        
        all_bb_emb = embed_bbs(bb_encoder, all_bbs_flat, DEVICE)
        
        # Synchronize CUDA to catch any delayed errors
        if torch.cuda.is_available():
            torch.cuda.synchronize()
        
        # Assign embeddings back to each sample
        cg_data_list = []
        offset = 0
        for i, (raw_data, n_bbs) in enumerate(zip(batch, batch_n_bbs)):
            bb_emb = all_bb_emb[offset:offset + n_bbs]
            offset += n_bbs
            cg_data = prepare_cg_data(raw_data, bb_emb)
            cg_data_list.append(cg_data)
        
        loader = DataLoader(cg_data_list, batch_size=len(cg_data_list), shuffle=False)
        b = next(iter(loader)).to(DEVICE)
        
        print(f"  CG batch: num_atoms={b.num_atoms}, frac_coords shape={b.frac_coords.shape}, atom_types shape={b.atom_types.shape}")
        
        try:
            with torch.no_grad():
                z = model.encode(b)[2]  # latent z, consistent with optimize.py
            
            # Synchronize CUDA
            if torch.cuda.is_available():
                torch.cuda.synchronize()
            
            z_chunks.append(z.detach().cpu())
            y_list.extend(batch_y)
            id_list.extend(batch_id)
        except Exception as e:
            print(f"Error in CG model encode: {e}")
            print(f"  b.num_atoms: {b.num_atoms}")
            raise
        
        batch, batch_y, batch_id, batch_bbs, batch_n_bbs = [], [], [], [], []

    total = 0
    kept = 0
    skipped_elements = 0
    skipped_bb = 0

    with env.begin() as txn:
        for k, v in tqdm(txn.cursor(), desc="Reading LMDB"):
            total += 1
            d = pickle.loads(v)

            pd = getattr(d, "prop_dict", None)
            if not isinstance(pd, dict) or PROP_KEY not in pd:
                continue

            # minimal CG requirements
            if not (hasattr(d, "cg_edge_index") and hasattr(d, "cg_frac_coords") and hasattr(d, "bbs")):
                continue

            # Check if all elements are supported by the pretrained encoder
            if not check_elements_supported(d.bbs):
                skipped_elements += 1
                continue

            # Preprocess all BBs
            processed_bbs = []
            bb_valid = True
            for bb in d.bbs:
                processed_bb = preprocess_bb(bb)
                if processed_bb is None:
                    bb_valid = False
                    break
                processed_bbs.append(processed_bb)
            
            if not bb_valid or not processed_bbs:
                skipped_bb += 1
                continue

            y = float(pd[PROP_KEY])
            mid = getattr(d, "m_id", None)

            batch.append(d)
            batch_y.append(y)
            batch_id.append(mid)
            batch_bbs.append(processed_bbs)
            batch_n_bbs.append(len(processed_bbs))
            kept += 1

            if len(batch) >= 16:
                flush()

    flush()

    if not z_chunks:
        print("No valid samples found!")
        return

    Z = torch.cat(z_chunks, dim=0)
    Y = torch.tensor(y_list, dtype=torch.float32).view(-1, 1)

    torch.save({"z": Z, "y": Y, "m_id": id_list}, OUT_PATH)
    print("Total LMDB records:", total)
    print("Kept for training:", kept)
    print("Skipped (unsupported elements):", skipped_elements)
    print("Skipped (invalid BBs):", skipped_bb)
    print("Saved:", OUT_PATH)
    print("Z shape:", Z.shape, "Y shape:", Y.shape)

if __name__ == "__main__":
    main()
