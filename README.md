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
import psutil  # Add this to monitor memory usage (pip install psutil)

warnings.filterwarnings("ignore")

def log_memory_usage(message):
    """Log memory usage for debugging"""
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    print(f"MEMORY [{message}]: {memory_info.rss / (1024 * 1024):.1f} MB")

def detect_bpm(y, sr, genre=None):
    try:
        log_memory_usage(f"detect_bpm start - array shape: {y.shape}, sr: {sr}")
        
        # Make sure were working with mono audio
        if y.ndim > 1:
            y = np.mean(y, axis=0)
        
        # Initialize BPM candidates list
        bpm_candidates = []
        
        print(f"Starting BPM detection with {len(y)} samples")
        
        # Method 1: Standard beat_track with different hop lengths
        for hop_length in [512, 256, 1024]:
            try:
                print(f"Trying beat_track with hop_length={hop_length}")
                tempo, _ = librosa.beat.beat_track(y=y, sr=sr, hop_length=hop_length)
                if np.isscalar(tempo) and 20 <= tempo <= 200:
                    bpm_candidates.append(tempo)
                print(f"beat_track result: {tempo}")
            except Exception as e:
                print(f"beat_track error with hop_length={hop_length}: {e}")
                pass
        
        # Method 2: Using onset strength
        try:
            print("Trying onset strength method")
            onset_env = librosa.onset.onset_strength(y=y, sr=sr)
            tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
            if np.isscalar(tempo) and 20 <= tempo <= 200:
                bpm_candidates.append(tempo)
            print(f"onset strength result: {tempo}")
        except Exception as e:
            print(f"onset strength error: {e}")
            pass
        
        # Method 3: Harmonic percussive source separation for clearer beat detection
        try:
            print("Trying HPSS method")
            y_harmonic, y_percussive = librosa.effects.hpss(y)
            tempo = librosa.beat.tempo(y=y_percussive, sr=sr)[0]
            if np.isscalar(tempo) and 20 <= tempo <= 200:
                bpm_candidates.append(tempo)
            print(f"HPSS result: {tempo}")
        except Exception as e:
            print(f"HPSS error: {e}")
            pass
        
        # Method 4: Using different onset detection methods
        onset_methods = ['librosa', 'superflux', 'spectral']
        for method in onset_methods:
            try:
                print(f"Trying onset detection with method={method}")
                onset_env = librosa.onset.onset_strength(y=y, sr=sr, feature=method)
                tempo = librosa.beat.tempo(onset_envelope=onset_env, sr=sr)[0]
                if np.isscalar(tempo) and 20 <= tempo <= 200:
                    bpm_candidates.append(tempo)
                print(f"onset detection result with {method}: {tempo}")
            except Exception as e:
                print(f"onset detection error with {method}: {e}")
                pass
        
        print(f"BPM candidates: {bpm_candidates}")
        
        # If we have candidates use median to get robust estimate
        if bpm_candidates:
            # Find the most common BPM range cluster
            bpm_ranges = {}
            for bpm in bpm_candidates:
                bpm_range = (bpm // 5) * 5  # Cluster within 5 BPM
                bpm_ranges[bpm_range] = bpm_ranges.get(bpm_range, 0) + 1
            
            print(f"BPM ranges: {bpm_ranges}")
            
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
            
            print(f"No candidates found, using default for genre {genre}: {bpm}")
        
        log_memory_usage(f"detect_bpm end - result: {bpm}")
        
        # Round to nearest integer
        return int(round(bpm))
    except Exception as e:
        print(f"Unexpected error in detect_bpm: {e}")
        print(traceback.format_exc())
        if genre and 'ambient' in genre.lower():
            return 70
        else:
            return 120

def analyze_track(file_path):
    try:
        print(f"\nProcessing file: {file_path}")
        log_memory_usage(f"analyze_track start - {os.path.basename(file_path)}")
        
        # Get the filename
        filename = os.path.basename(file_path)
        
        # Skip macOS hidden files completely
        if filename.startswith('._'):
            print(f"Skipping hidden file: {filename}")
            return None
            
        # Extract genre from path for better BPM defaults
        path_parts = file_path.split(os.sep)
        genre = None
        for part in path_parts:
            if part in ['ambient', 'downtempo', 'psytrance', 'drum-n-bass', 'psy-wubs', 'psy-uptempo']: # Again I should probably come back and generalize this later...
                genre = part
                break
        
        print(f"Detected genre: {genre}")
        bpm = None
            
        # Load the audio file
        if file_path.lower().endswith('.flac'):
            print(f"Processing FLAC file: {filename}")
            # Try to get metadata from FLAC tags
            try:
                audio = FLAC(file_path)
                # Try to get BPM from metadata first
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
            
            # If metadata doesnt have BPM analyze the audio
            if not bpm:
                try:
                    print(f"No BPM in metadata, analyzing audio...")
                    # Only analyze first minute for speed
                    duration = 30 if sys.platform == 'linux' else 60
                    
                    # Always ensure mono loading to avoid array shape issues
                    print(f"Loading audio with duration={duration}s, mono=True")
                    y, sr = librosa.load(file_path, sr=22050, duration=duration, mono=True)  # Lowered sample rate
                    log_memory_usage(f"After librosa.load - array shape: {y.shape}")
                    
                    # Truncate y if its somehow not mono?
                    if y.ndim > 1:
                        print(f"Unexpected: audio still has {y.ndim} dimensions after mono loading")
                        y = np.mean(y, axis=0)
                    
                    # Get BPM with robust detection
                    print(f"Detecting BPM...")
                    bpm = detect_bpm(y, sr, genre)
                    print(f"Detected BPM: {bpm}")
                    
                    # Force cleanup
                    del y
                    gc.collect()
                    log_memory_usage("After audio analysis and cleanup")
                    
                except Exception as e:
                    print(f"Error analyzing audio: {e}")
                    print(traceback.format_exc())
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
                    print(f"Using default BPM for genre: {bpm}")
            
            # Round BPM to nearest integer
            bpm = int(round(float(bpm)))
        
        elif file_path.lower().endswith('.wav'):
            print(f"Processing WAV file: {filename}")
            # Similar process for WAV files
            try:
                print(f"Loading audio...")
                # Use shorter duration on Linux
                duration = 30 if sys.platform == 'linux' else 60
                y, sr = librosa.load(file_path, sr=22050, duration=duration, mono=True)  # Lowered sample rate
                
                # Make sure it's mono
                if y.ndim > 1:
                    print(f"Converting {y.ndim} dimensions to mono")
                    y = np.mean(y, axis=0)
                
                # Get BPM with robust detection
                print(f"Detecting BPM...")
                bpm = detect_bpm(y, sr, genre)
                print(f"Detected BPM: {bpm}")
                
                # Force cleanup
                del y
                gc.collect()
                
            except Exception as e:
                print(f"Error processing WAV: {e}")
                print(traceback.format_exc())
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
                print(f"Using default BPM for genre: {bpm}")
        else:
            print(f"Unsupported file format: {filename}")
            return None
        
        log_memory_usage(f"analyze_track end - {filename}")    
        return {
            'file_path': file_path,
            'bpm': bpm,
            'filename': filename
        }
    except Exception as e:
        print(f"Unexpected error in analyze_track: {e}")
        print(traceback.format_exc())
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
        print(f"Error in process_file: {e}")
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

def organize_music_library(source_dir, target_dir, num_processes=None):
    # For Linux platform, use sequential processing
    if sys.platform == 'linux':
        print("Linux platform detected: using sequential processing")
        return organize_music_library_linux(source_dir, target_dir)
    
    # For macOS and other platforms, use multiprocessing as before
    if num_processes is None:
        num_processes = max(1, multiprocessing.cpu_count() - 1)
    
    print(f"Using {num_processes} CPU cores for processing")
    
    # Get list of audio files
    print("Finding audio files...")
    audio_files = find_audio_files(source_dir)
    
    # Filter out any hidden files
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('._')]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Create multiprocessing pool
    pool = multiprocessing.Pool(processes=num_processes)
    
    # Analyze all tracks in parallel
    print("Analyzing audio files for BPM...")
    start_time = time.time()
    
    # Use imap instead of tqdm for better debugging
    results = []
    for i, result in enumerate(pool.imap(analyze_track, audio_files)):
        results.append(result)
        print(f"Processed {i+1}/{len(audio_files)} files")
    
    # Filter out None results
    results = [r for r in results if r is not None] 
    
    # Process files with analyzed data
    print("Organizing files by BPM...")
    process_func = partial(process_file, source_dir=source_dir, target_dir=target_dir)
    
    # Use imap instead of tqdm
    organized_results = []
    for i, result in enumerate(pool.imap(process_func, results)):
        organized_results.append(result)
        print(f"Organized {i+1}/{len(results)} files")
    
    # Close the pool
    pool.close()
    pool.join()
    
    # Calculate statistics
    success_count = sum(1 for r in organized_results if r['status'] == 'success')
    error_count = sum(1 for r in organized_results if r['status'] == 'error')
    skipped_count = sum(1 for r in organized_results if r['status'] == 'skipped')
    
    # Print summary
    elapsed_time = time.time() - start_time
    print(f"\nProcessing complete in {elapsed_time:.2f} seconds!")
    print(f"Successfully processed: {success_count} files")
    print(f"Errors encountered: {error_count} files")
    print(f"Files skipped: {skipped_count} files")
    
    # Print errors if any
    if error_count > 0:
        print("\nErrors:")
        for r in organized_results:
            if r['status'] == 'error':
                print(f"  {r['file']}: {r['error']}")
    
    # Show BPM ranges detected
    bpm_ranges = {}
    for r in organized_results:
        if r['status'] == 'success':
            bpm_floor = (r['bpm'] // 10) * 10
            bpm_range = f"{bpm_floor}-{bpm_floor + 10}"
            bpm_ranges[bpm_range] = bpm_ranges.get(bpm_range, 0) + 1
    
    print("\nBPM Distribution:")
    for bpm_range, count in sorted(bpm_ranges.items()):
        print(f"  {bpm_range} BPM: {count} tracks")

def organize_music_library_linux(source_dir, target_dir):
    """Simple sequential processing for Linux systems to avoid memory issues"""
    print("Using sequential processing for Linux")
    
    # Get list of audio files
    print("Finding audio files...")
    audio_files = find_audio_files(source_dir)
    
    # Filter out any hidden files
    audio_files = [f for f in audio_files if not os.path.basename(f).startswith('._')]
    
    print(f"Found {len(audio_files)} audio files to process")
    
    # Analyze all tracks sequentially with just genre-based defaults
    print("Processing files with genre-based BPM defaults...")
    start_time = time.time()
    
    all_results = []
    for i, file_path in enumerate(audio_files):
        try:
            print(f"\nProcessing file {i+1}/{len(audio_files)}: {os.path.basename(file_path)}")
            
            # Extract genre for default BPM
            path_parts = file_path.split(os.sep)
            genre = None
            for part in path_parts:
                if part in ['ambient', 'downtempo', 'psytrance', 'drum-n-bass', 'psy-wubs', 'psy-uptempo']:
                    genre = part
                    break
            
            # Skip BPM detection on Linux, use genre defaults to avoid segfaults
            if genre and 'ambient' in genre.lower():
                bpm = 70
            elif genre and 'downtempo' in genre.lower():
                bpm = 90
            elif genre and 'psytrance' in genre.lower():
                bpm = 145
            elif genre and 'drum-n-bass' in genre.lower():
                bpm = 170
            elif genre and 'psy-wubs' in genre.lower():
                bpm = 110
            elif genre and 'psy-uptempo' in genre.lower():
                bpm = 130
            else:
                bpm = 120
            
            filename = os.path.basename(file_path)
            
            # Create result dict with file info
            result = {
                'file_path': file_path,
                'bpm': bpm,
                'filename': filename
            }
            
            # Now use the process_file function to organize properly by genre and BPM
            organized_result = process_file(result, source_dir, target_dir)
            all_results.append(organized_result)
            
            print(f"Organized {filename} (Genre: {extract_genre_from_path(file_path, source_dir)}, BPM: {bpm})")
            
            # Force garbage collection
            gc.collect()
            
        except Exception as e:
            print(f"Error processing {os.path.basename(file_path)}: {e}")
            print(traceback.format_exc())
    
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