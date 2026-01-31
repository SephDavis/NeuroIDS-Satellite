"""
NSL-KDD Data Loader for NeuroIDS-Sat

Handles loading and preprocessing of the NSL-KDD dataset
with satellite-specific adaptations.

Dataset available at: https://www.unb.ca/cic/datasets/nsl.html

Author: Toby R. Davis
"""

import numpy as np
from pathlib import Path
from typing import Tuple, Dict, Optional
import warnings


# NSL-KDD column names
COLUMN_NAMES = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes',
    'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in',
    'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations',
    'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login',
    'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate',
    'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate',
    'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count',
    'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
    'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate',
    'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'attack_type', 'difficulty'
]

# Attack type to category mapping
ATTACK_CATEGORIES = {
    'normal': 0,
    # DoS attacks
    'back': 1, 'land': 1, 'neptune': 1, 'pod': 1, 'smurf': 1, 'teardrop': 1,
    'mailbomb': 1, 'apache2': 1, 'processtable': 1, 'udpstorm': 1,
    # Probe attacks  
    'ipsweep': 2, 'nmap': 2, 'portsweep': 2, 'satan': 2, 'mscan': 2, 'saint': 2,
    # R2L attacks
    'ftp_write': 3, 'guess_passwd': 3, 'imap': 3, 'multihop': 3, 'phf': 3,
    'spy': 3, 'warezclient': 3, 'warezmaster': 3, 'sendmail': 3, 'named': 3,
    'snmpgetattack': 3, 'snmpguess': 3, 'xlock': 3, 'xsnoop': 3, 'worm': 3,
    # U2R attacks
    'buffer_overflow': 4, 'loadmodule': 4, 'perl': 4, 'rootkit': 4, 'httptunnel': 4,
    'ps': 4, 'sqlattack': 4, 'xterm': 4
}

# Categorical columns that need encoding
CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']

# Protocol types
PROTOCOLS = {'tcp': 0, 'udp': 1, 'icmp': 2}

# Service types (70 unique services in NSL-KDD)
SERVICES = {
    'http': 0, 'smtp': 1, 'finger': 2, 'domain_u': 3, 'auth': 4, 'telnet': 5,
    'ftp': 6, 'eco_i': 7, 'ntp_u': 8, 'ecr_i': 9, 'other': 10, 'private': 11,
    'pop_3': 12, 'ftp_data': 13, 'rje': 14, 'time': 15, 'mtp': 16, 'link': 17,
    'remote_job': 18, 'gopher': 19, 'ssh': 20, 'name': 21, 'whois': 22,
    'domain': 23, 'login': 24, 'imap4': 25, 'daytime': 26, 'ctf': 27, 'nntp': 28,
    'shell': 29, 'IRC': 30, 'nnsp': 31, 'http_443': 32, 'exec': 33, 'printer': 34,
    'efs': 35, 'courier': 36, 'uucp': 37, 'klogin': 38, 'kshell': 39, 'echo': 40,
    'discard': 41, 'systat': 42, 'supdup': 43, 'iso_tsap': 44, 'hostnames': 45,
    'csnet_ns': 46, 'pop_2': 47, 'sunrpc': 48, 'uucp_path': 49, 'netbios_ns': 50,
    'netbios_ssn': 51, 'netbios_dgm': 52, 'sql_net': 53, 'vmnet': 54, 'bgp': 55,
    'Z39_50': 56, 'ldap': 57, 'netstat': 58, 'urh_i': 59, 'X11': 60, 'urp_i': 61,
    'pm_dump': 62, 'tftp_u': 63, 'tim_i': 64, 'red_i': 65, 'icmp': 66,
    'http_2784': 67, 'harvest': 68, 'aol': 69
}

# Flag types
FLAGS = {
    'SF': 0, 'S0': 1, 'REJ': 2, 'RSTR': 3, 'RSTO': 4, 'SH': 5, 'S1': 6,
    'S2': 7, 'RSTOS0': 8, 'S3': 9, 'OTH': 10
}


def load_nslkdd(data_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load NSL-KDD dataset.
    
    Args:
        data_path: Path to directory containing KDDTrain+.txt and KDDTest+.txt
        
    Returns:
        X_train, y_train, X_test, y_test
    """
    data_path = Path(data_path)
    
    train_file = data_path / 'KDDTrain+.txt'
    test_file = data_path / 'KDDTest+.txt'
    
    if not train_file.exists():
        raise FileNotFoundError(
            f"Training file not found: {train_file}\n"
            f"Download from: https://www.unb.ca/cic/datasets/nsl.html"
        )
    
    if not test_file.exists():
        raise FileNotFoundError(
            f"Test file not found: {test_file}\n"
            f"Download from: https://www.unb.ca/cic/datasets/nsl.html"
        )
    
    print(f"Loading training data from {train_file}...")
    X_train, y_train = _load_file(train_file)
    
    print(f"Loading test data from {test_file}...")
    X_test, y_test = _load_file(test_file)
    
    return X_train, y_train, X_test, y_test


def _load_file(filepath: Path) -> Tuple[np.ndarray, np.ndarray]:
    """Load a single NSL-KDD file."""
    
    X_list = []
    y_list = []
    
    with open(filepath, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            
            if len(parts) < 42:
                continue
            
            # Extract features
            features = []
            
            # Numeric features (indices 0, 4-40)
            for i in [0] + list(range(4, 41)):
                try:
                    features.append(float(parts[i]))
                except (ValueError, IndexError):
                    features.append(0.0)
            
            # Categorical features (indices 1, 2, 3)
            # Protocol type
            protocol = parts[1].lower()
            features.append(PROTOCOLS.get(protocol, len(PROTOCOLS)))
            
            # Service
            service = parts[2].lower()
            features.append(SERVICES.get(service, len(SERVICES)))
            
            # Flag
            flag = parts[3].upper()
            features.append(FLAGS.get(flag, len(FLAGS)))
            
            X_list.append(features)
            
            # Extract label
            attack_type = parts[41].lower().replace('.', '')
            label = ATTACK_CATEGORIES.get(attack_type, 0)
            y_list.append(label)
    
    X = np.array(X_list, dtype=np.float32)
    y = np.array(y_list, dtype=np.int32)
    
    return X, y


def preprocess_data(X_train: np.ndarray, X_test: np.ndarray,
                    normalize: bool = True) -> Tuple[np.ndarray, np.ndarray]:
    """Preprocess features for SNN input.
    
    Args:
        X_train: Training features
        X_test: Test features
        normalize: Whether to normalize to [0, 1]
        
    Returns:
        Preprocessed X_train, X_test
    """
    X_train = X_train.copy()
    X_test = X_test.copy()
    
    if normalize:
        # Compute statistics on training data
        X_min = X_train.min(axis=0)
        X_max = X_train.max(axis=0)
        X_range = X_max - X_min
        
        # Avoid division by zero
        X_range[X_range == 0] = 1.0
        
        # Normalize both sets using training statistics
        X_train = (X_train - X_min) / X_range
        X_test = (X_test - X_min) / X_range
        
        # Clip to [0, 1]
        X_train = np.clip(X_train, 0, 1)
        X_test = np.clip(X_test, 0, 1)
    
    return X_train, X_test


def create_balanced_sample(X: np.ndarray, y: np.ndarray,
                           max_per_class: int = 5000,
                           min_per_class: int = 100) -> Tuple[np.ndarray, np.ndarray]:
    """Create a balanced sample of the dataset.
    
    For satellite paper experiments, we balance to avoid
    majority class bias while keeping minority classes.
    
    Args:
        X: Features
        y: Labels
        max_per_class: Maximum samples per class (undersample majority)
        min_per_class: Minimum samples per class (oversample minority)
        
    Returns:
        Balanced X, y
    """
    X_balanced = []
    y_balanced = []
    
    classes = np.unique(y)
    
    for c in classes:
        mask = y == c
        X_c = X[mask]
        
        n_samples = len(X_c)
        
        if n_samples > max_per_class:
            # Undersample
            idx = np.random.choice(n_samples, max_per_class, replace=False)
            X_c = X_c[idx]
        elif n_samples < min_per_class:
            # Oversample with slight noise
            n_needed = min_per_class - n_samples
            idx = np.random.choice(n_samples, n_needed, replace=True)
            X_extra = X_c[idx] + np.random.randn(n_needed, X_c.shape[1]) * 0.01
            X_c = np.vstack([X_c, X_extra])
        
        X_balanced.append(X_c)
        y_balanced.extend([c] * len(X_c))
    
    X_balanced = np.vstack(X_balanced)
    y_balanced = np.array(y_balanced)
    
    # Shuffle
    idx = np.random.permutation(len(y_balanced))
    
    return X_balanced[idx], y_balanced[idx]


def get_dataset_statistics(y: np.ndarray) -> Dict:
    """Get dataset class distribution statistics."""
    
    unique, counts = np.unique(y, return_counts=True)
    total = len(y)
    
    category_names = ['normal', 'dos', 'probe', 'r2l', 'u2r']
    
    stats = {
        'total': total,
        'classes': {}
    }
    
    for c, count in zip(unique, counts):
        name = category_names[c] if c < len(category_names) else f'class_{c}'
        stats['classes'][name] = {
            'count': int(count),
            'percentage': count / total * 100
        }
    
    return stats


if __name__ == "__main__":
    # Test the data loader
    import sys
    
    if len(sys.argv) > 1:
        data_path = sys.argv[1]
    else:
        data_path = 'data/NSL-KDD/'
    
    try:
        X_train, y_train, X_test, y_test = load_nslkdd(data_path)
        X_train, X_test = preprocess_data(X_train, X_test)
        
        print(f"\nDataset loaded successfully!")
        print(f"Training samples: {len(X_train):,}")
        print(f"Test samples: {len(X_test):,}")
        print(f"Features: {X_train.shape[1]}")
        
        print("\nTraining set statistics:")
        stats = get_dataset_statistics(y_train)
        for name, info in stats['classes'].items():
            print(f"  {name}: {info['count']:,} ({info['percentage']:.1f}%)")
        
        print("\nTest set statistics:")
        stats = get_dataset_statistics(y_test)
        for name, info in stats['classes'].items():
            print(f"  {name}: {info['count']:,} ({info['percentage']:.1f}%)")
            
    except FileNotFoundError as e:
        print(f"Error: {e}")
        print("\nTo download the dataset:")
        print("1. Visit https://www.unb.ca/cic/datasets/nsl.html")
        print("2. Download KDDTrain+.txt and KDDTest+.txt")
        print("3. Place them in the data/NSL-KDD/ directory")
