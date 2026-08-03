import os
import shutil
import random
from tqdm import tqdm

def split_aligned_dataset(source_img_dir, source_mask_dir, dest_dir, train_ratio=0.8, val_ratio=0.1, test_ratio=0.1, seed=42):
    random.seed(seed)
    assert abs((train_ratio + val_ratio + test_ratio) - 1.0) < 1e-9, "Le percentuali devono sommare a 1.0"
    
    # 1. Leggi i file ed escludi file nascosti di sistema
    img_files = sorted([f for f in os.listdir(source_img_dir) if not f.startswith('.')])
    mask_files = sorted([f for f in os.listdir(source_mask_dir) if not f.startswith('.')])
    
    # 2. Crea mappature dei nomi senza estensione
    img_map = {os.path.splitext(f)[0]: f for f in img_files}
    mask_map = {os.path.splitext(f)[0]: f for f in mask_files}
    
    # Trova l'intersezione (i file che hanno sia immagine che maschera)
    aligned_names = sorted(list(set(img_map.keys()) & set(mask_map.keys())))
    
    total_aligned = len(aligned_names)
    print(f"📈 File totali in cartella immagini: {len(img_files)}")
    print(f"📈 File totali in cartella maschere: {len(mask_files)}")
    print(f"✅ Coppie perfettamente allineate trovate: {total_aligned}")
    
    # Segnala quanti file sono rimasti orfani
    orphans_img = len(img_files) - total_aligned
    orphans_mask = len(mask_files) - total_aligned
    if orphans_img > 0 or orphans_mask > 0:
        print(f"⚠️ Attenzione: Verranno ignorati {orphans_img} file immagine e {orphans_mask} file maschera perché spaiati.")

    # 3. Mescola ed esegui lo split solo sui file allineati
    random.shuffle(aligned_names)
    
    train_end = int(total_aligned * train_ratio)
    val_end = train_end + int(total_aligned * val_ratio)
    
    splits = {
        'train': aligned_names[:train_end],
        'val': aligned_names[train_end:val_end],
        'test': aligned_names[val_end:]
    }
    
    # 4. Copia i file nelle cartelle di destinazione
    for split_name, names_list in splits.items():
        img_dest = os.path.join(dest_dir, split_name, 'images')
        mask_dest = os.path.join(dest_dir, split_name, 'masks')
        
        # Rimuove la cartella se esisteva già da un tentativo precedente, per evitare duplicati
        if os.path.exists(img_dest):
            shutil.rmtree(img_dest)
        if os.path.exists(mask_dest):
            shutil.rmtree(mask_dest)
            
        os.makedirs(img_dest, exist_ok=True)
        os.makedirs(mask_dest, exist_ok=True)
        
        print(f"\n🚚 Copia in corso per lo split {split_name.upper()} ({len(names_list)} coppie)...")
        for name in tqdm(names_list):
            img_file = img_map[name]
            mask_file = mask_map[name]
            
            src_img = os.path.join(source_img_dir, img_file)
            src_mask = os.path.join(source_mask_dir, mask_file)
            
            shutil.copy(src_img, os.path.join(img_dest, img_file))
            shutil.copy(src_mask, os.path.join(mask_dest, mask_file))

    print("\n🎉 Split completato con successo e file allineati correttamente!")

if __name__ == "__main__":
    # Configura i tuoi percorsi reali locali qui
    SOURCE_IMAGES = "C:/Users/Mannino/Desktop/Università/Computer Vision/sample_dataset/images" 
    SOURCE_MASKS = "C:/Users/Mannino/Desktop/Università/Computer Vision/sample_dataset/masks"
    DESTINATION_DATA = "data"
    
    split_aligned_dataset(
        source_img_dir=SOURCE_IMAGES,
        source_mask_dir=SOURCE_MASKS,
        dest_dir=DESTINATION_DATA,
        train_ratio=0.8,
        val_ratio=0.1,
        test_ratio=0.1
    )