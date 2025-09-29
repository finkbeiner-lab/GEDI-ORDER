#!/usr/bin/env python3

import tensorflow as tf
import subprocess
import sys

def check_cudnn_version():
    """Check CuDNN version from library files"""
    try:
        result = subprocess.run(['strings', '/usr/local/cuda-11.8/lib64/libcudnn.so.8'],
                              capture_output=True, text=True)
        for line in result.stdout.split('\n'):
            if 'CUDNN_VERSION' in line:
                print(f"CuDNN library version: {line}")
                break
    except Exception as e:
        print(f"Could not check CuDNN version: {e}")

def test_tensorflow():
    """Test TensorFlow GPU functionality"""
    print(f"TensorFlow version: {tf.__version__}")
    print(f"Built with CUDA: {tf.test.is_built_with_cuda()}")

    gpus = tf.config.list_physical_devices('GPU')
    print(f"GPUs available: {len(gpus)}")

    if gpus:
        print("GPU details:")
        for gpu in gpus:
            print(f"  {gpu}")

        # Test a simple operation
        try:
            with tf.device('/GPU:0'):
                a = tf.constant([[1.0, 2.0], [3.0, 4.0]])
                b = tf.constant([[1.0, 1.0], [0.0, 1.0]])
                c = tf.matmul(a, b)
                print("GPU computation test: SUCCESS")
                print(f"Result: {c.numpy()}")
        except Exception as e:
            print(f"GPU computation test: FAILED - {e}")
            return False
    else:
        print("No GPUs detected")
        return False

    return True

if __name__ == "__main__":
    print("=== CuDNN and TensorFlow Verification ===")
    check_cudnn_version()
    print()
    success = test_tensorflow()

    if success:
        print("\n✅ CuDNN upgrade successful! GPU acceleration is working.")
        sys.exit(0)
    else:
        print("\n❌ Issues detected with GPU setup.")
        sys.exit(1)