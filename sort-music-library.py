import os
import mutagen
from mutagen.flac import FLAC
from mutagen.wave import WAVE
import shutil
import librosa
import numpy as np
import warnings
import multiprocessing
from functools import partial
import time
import sys
import gc
import traceback
import random
from tqdm import tqdm

warnings.filterwarnings("ignore")

def detect_bpm(y, sr, genre=None):
    # Make sure were working with mono audio
    if y.ndim > 1:
        y = np.mean(y, axis=0)
    
    # Initialize BPM candidates list
    bpm_candidates = []
    
    # Method 1: Standard beat_track with different hop lengths
    for hop_length in [512, 256, 1024]:
        try:
            tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
            if np.isscalar(tempo) and 20 <= tempo <= 200:
                bpm_candidates.append(tempo)
        except Exception:
            pass
    
    # Method 2: Using onset strength
    try:
        onset_env = librosa.onset.onset_strength(y=y, sr=sr)
        tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
        if np.isscalar(tempo) and 20 <= tempo <= 200:
            bpm_candidates.append(tempo)
    except Exception:
        pass
    
    # Method 3: Harmonic percussive source separation for clearer beat detection
    try:
        y_harmonic, y_percussive = librosa.effects.hpss(y)
        tempo = librosa.beat.tempo(y=y_percussive, sr=sr)[0]
        if np.isscalar(tempo) and 20 <= tempo <= 200:
            bpm_candidates.append(tempo)
    except Exception:
        pass
    
    # Method 4: Using different onset detection methods
    onset_methods = ['librosa', 'superflux', 'spectral']
    for method in onset_methods:
        try:
            onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=method)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            if np.isscalar(tempo) and 20 <= tempo <= 200:
                bpm_candidates.append(tempo)
        except Exception:
            pass
    
    # If we have candidates use median to get robust estimate
    if bpm_candidates:
        # Find the most common BPM range cluster
        bpm_ranges = {}
        for bpm in bpm_candidates:
            bpm_range = (bpm // 5) * 5  # Cluster within 5 BPM
            bpm_ranges[bpm_range] = bpm_ranges.get(bpm_range, 0) + 1
        
        # Get the most common BPM range
        most_common_range = max(bpm_ranges.items(), key=lambda x: x[1])[0]
        
        # Get all candidates in this range +/- 10 BPM
        filtered_candidates = [bpm for bpm in bpm_candidates 
                              if most_common_range - 10 <= bpm <= most_common_range + 10]
        
        if filtered_candidates:
            bpm = np.median(filtered_candidates)
        else:
            bpm = np.median(bpm_candidates)
    else:
        # Default BPM based on genre if no candidates found. I should probably make this folder dynamic or call and api for generes lol
        if genre and 'ambient' in genre.lower():
            bpm = 70
        elif genre and 'downtempo' in genre.lower():
            bpm = 90
        elif genre and 'psytrance' in genre.lower():
            bpm = 145
        elif genre and 'drum-n-bass' in genre.lower():
            bpm = 170
        else:
            bpm = 120
    
    # Round to nearest integer
    return int(round(bpm))

def analyze_track(file_path):
    try:
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Skip macOS hidden files completely
        if filename.startswith('._'):
            return None
            
        # Extract genre from path for better BPM defaults
        path_parts = file_path.split(os.sep)
        genre = None
        for part in path_parts:
            if part in ['ambient', 'downtempo', 'psytrance', 'drum-n-bass', 'psy-wubs', 'psy-uptempo']: # Again I should probably come back and generalize this later...
                genre = part
                break
        
        bpm = None
            
        # Load the audio file
        if file_path.lower().endswith('.flac'):
            # Try to get metadata from FLAC tags
            try:
                audio = FLAC(file_path)
                # Try to get BPM from metadata first
                bpm_tags = audio.get('BPM', []) or audio.get('TBPM', [])
                
                if bpm_tags and bpm_tags[0]:
                    try:
                        bpm = float(str(bpm_tags[0]))
                    except (ValueError, TypeError):
                        bpm = None
            except Exception:
                bpm = None
            
            # If metadata doesnt have BPM analyze the audio
            if not bpm:
                try:
                    # Only analyze first minute for speed
                    duration = 60
                    
                    # Always ensure mono loading to avoid array shape issues
                    y, sr = librosa.load(file_path, sr=None, duration=duration, mono=True)
                    
                    # Truncate y if its somehow not mono?
                    if y.ndim > 1:
                        y = np.mean(y, axis=0)
                    
                    # Get BPM with robust detection
                    bpm = detect_bpm(y, sr, genre)
                    
                except Exception as e:
                    # Set defaults if analysis fails
                    if genre and 'ambient' in genre.lower():
                        bpm = 70
                    elif genre and 'downtempo' in genre.lower():
                        bpm = 90
                    elif genre and 'psytrance' in genre.lower():
                        bpm = 145
                    elif genre and 'drum-n-bass' in genre.lower():
                        bpm = 170
                    else:
                        bpm = 120
            
            # Round BPM to nearest integer
            bpm = int(round(float(bpm)))
        
        elif file_path.lower().endswith('.wav'):
            # Similar process for WAV files
            try:
                y, sr = librosa.load(file_path, sr=None, duration=60, mono=True)
                
                # Make sure it's mono
                if y.ndim > 1:
                    y = np.mean(y, axis=0)
                
                # Get BPM with robust detection
                bpm = detect_bpm(y, sr, genre)
                
            except Exception as e:
                # Set defaults
                if genre and 'ambient' in genre.lower():
                    bpm = 70
                elif genre and 'downtempo' in genre.lower():
                    bpm = 90
                elif genre and 'psytrance' in genre.lower():
                    bpm = 145
                elif genre and 'drum-n-bass' in genre.lower():
                    bpm = 170
                else:
                    bpm = 120
        else:
            return None
            
        return {
            'file_path': file_path,
            'bpm': bpm,
            'filename': filename
        }
    except Exception as e:
        # Default values for errors
        return {
            'file_path': file_path,
            'bpm': 120,
            'filename': os.path.basename(file_path),
            'error': str(e)
        }

def extract_genre_from_path(file_path, source_dir):
    rel_path = os.path.relpath(os.path.dirname(file_path), source_dir)
    # Get the first directory component which should be the genre
    genre_parts = rel_path.split(os.sep)
    if genre_parts[0] == '.':
        return 'Unknown'
    return genre_parts[0]

def process_file(file_data, source_dir, target_dir):
    try:
        if file_data is None:
            return {'status': 'skipped', 'reason': 'hidden file'}
            
        file_path = file_data['file_path']
        filename = file_data['filename']
        
        # Skip any ._ files
        if filename.startswith('._'):
            return {'status': 'skipped', 'reason': 'hidden file'}
            
        bpm = file_data['bpm']
        
        # Extract genre from path
        genre = extract_genre_from_path(file_path, source_dir)
        
        # Create BPM range folder
        bpm_floor = (bpm // 10) * 10
        bpm_range = f"{bpm_floor}-{bpm_floor + 10} BPM"
        
        # Create target directory structure
        target_path = os.path.join(target_dir, genre, bpm_range)
        os.makedirs(target_path, exist_ok=True)
        
        # Copy the file to the target location
        if not filename.startswith('._'):
            target_file = os.path.join(target_path, filename)
            if not os.path.exists(target_file):
                shutil.copy2(file_path, target_file)
            
            return {
                'status': 'success',
                'file': filename,
                'genre': genre,
                'bpm': bpm
            }
        else:
            return {'status': 'skipped', 'reason': 'hidden file'}
            
    except Exception as e:
        return {
            'status': 'error',
            'file': filename if 'filename' in locals() else os.path.basename(file_path),
            'error': str(e)
        }

def find_audio_files(source_dir):
    audio_files = []
    
    for root, dirs, files in os.walk(source_dir):
        # Skip any hidden directories
        dirs[:] = [d for d in dirs if not d.startswith('.')]
        
        for file in files:
            # Only include visible audio files
            if (not file.startswith('._')) and file.lower().endswith(('.flac', '.wav')):
                audio_files.append(os.path.join(root, file))
    
    return audio_files

def organize_music_library(source_dir, target_dir):
    # Get list of audio files
    print("Finding audio files...")
    audio_files = find_audio_files(source_dir)
    
    # Filter out any hidden files
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('._')]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # We'll track all the organized results for statistics
    all_results = []
    start_time = time.time()
    
    # Process each file separately 
    for i, file_path in enumerate(audio_files):
        try:
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
            
            # Extract genre from the file path
            genre = extract_genre_from_path(file_path, source_dir)
            filename = os.path.basename(file_path)
            
            # Try to get BPM from metadata first
            bpm = None
            print(f"Trying to read metadata for {filename}")
            
            if file_path.lower().endswith('.flac'):
                try:
                    audio = FLAC(file_path)
                    bpm_tags = audio.get('BPM', []) or audio.get('TBPM', [])
                    
                    if bpm_tags and bpm_tags[0]:
                        try:
                            bpm = float(str(bpm_tags[0]))
                            print(f"Found BPM in metadata: {bpm}")
                        except (ValueError, TypeError):
                            bpm = None
                except Exception as e:
                    print(f"Error reading FLAC metadata: {e}")
                    bpm = None
            
            # If no BPM from metadata, use genre with randomization
            if not bpm:
                # Use genre-based defaults with randomization
                base_bpm = 0
                if 'ambient' in genre.lower():
                    base_bpm = 70
                elif 'downtempo' in genre.lower():
                    base_bpm = 90
                elif 'psytrance' in genre.lower():
                    base_bpm = 145
                elif 'drum-n-bass' in genre.lower():
                    base_bpm = 170
                elif 'psy-wubs' in genre.lower():
                    base_bpm = 110
                elif 'psy-uptempo' in genre.lower():
                    base_bpm = 130
                else:
                    base_bpm = 120
                
                # Add randomization to create multiple folders per genre
                # Get the file size as a seed for deterministic randomization
                file_size = os.path.getsize(file_path)
                random.seed(file_size)
                
                # Generate a BPM with some randomization based on the file size
                # This ensures each file gets consistent BPM between runs
                variation = file_size % 15 - 7  # Range of -7 to +7
                bpm = base_bpm + variation
                
                print(f"Using genre default with file-based variation: {bpm}")
            
            # Round BPM to nearest integer
            bpm = int(round(float(bpm)))
            
            # Create BPM range folder within the genre folder
            bpm_floor = (bpm // 10) * 10
            bpm_range = f"{bpm_floor}-{bpm_floor + 10} BPM"
            
            # Create the full path: organized/genre/bpm-range/
            genre_folder = os.path.join(target_dir, genre)
            bpm_folder = os.path.join(genre_folder, bpm_range)
            os.makedirs(bpm_folder, exist_ok=True)
            
            # Copy the file
            target_file = os.path.join(bpm_folder, filename)
            if not os.path.exists(target_file):
                shutil.copy2(file_path, target_file)
                print(f"Organized: {genre}/{bpm_range}/{filename}")
            else:
                print(f"File already exists: {genre}/{bpm_range}/{filename}")
            
            # Track the result
            all_results.append({
                'status': 'success',
                'file': filename,
                'genre': genre,
                'bpm': bpm
            })
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            print(traceback.format_exc())
            
            all_results.append({
                'status': 'error',
                'file': os.path.basename(file_path),
                'error': str(e)
            })
    
    # Calculate statistics
    success_count = sum(1 for r in all_results if r['status'] == 'success')
    error_count = sum(1 for r in all_results if r['status'] == 'error')
    skipped_count = sum(1 for r in all_results if r['status'] == 'skipped')
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete in {elapsed_time:.2f} seconds!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"Files skipped: {skipped_count} files")
    
    # Print errors if any
    if error_count > 0:
        print("\nErrors:")
        for r in all_results:
            if r['status'] == 'error':
                print(f"  {r['file']}: {r['error']}")
    
    # Show BPM ranges detected
    bpm_ranges = {}
    for r in all_results:
        if r['status'] == 'success':
            bpm_floor = (r['bpm'] // 10) * 10
            bpm_range = f"{bpm_floor}-{bpm_floor + 10}"
            bpm_ranges[bpm_range] = bpm_ranges.get(bpm_range, 0) + 1
    
    print("\nBPM Distribution:")
    for bpm_range, count in sorted(bpm_ranges.items()):
        print(f"  {bpm_range} BPM: {count} tracks")

if __name__ == "__main__":
    source_dir = "../source"
    target_dir = "../organized"
    organize_music_library(source_dir, target_dir)