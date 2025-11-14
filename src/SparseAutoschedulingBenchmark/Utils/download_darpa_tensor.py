"""
Script to download and convert DARPA tensor from FROSTT to binsparse format.
The DARPA tensor represents intrusion detection data with 28,436,033 non-zeros.
This is the smallest 3D tensor from FROSTT (exclusing matrix multiplication).

Prerequisites:
- Julia must be installed: https://julialang.org/downloads/
- Finch.jl package must be installed: julia -e 'using Pkg; Pkg.add("Finch")'
- h5py must be installed: pip install h5py
"""

import os
import subprocess
import sys
from pathlib import Path


def download_darpa_tensor(output_dir=None):
    """
    Download DARPA tensor from FROSTT and convert to binsparse format.

    Args:
        output_dir: Directory to save the tensor (defaults to repo_root/data)

    Returns:
        str: Path to the converted binsparse file, or none if failed
    """

    script_dir = Path(__file__).parent
    if output_dir is None:
        repo_root = script_dir.parent
        output_dir = repo_root / "data"

    output_dir = Path(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    print(f"Output directory: {output_dir}")

    url = "http://frostt.io/tensors/darpa/1998darpa.tns.gz"
    tns_gz_path = output_dir / "1998darpa.tns.gz"
    tns_path = output_dir / "1998darpa.tns"
    binsparse_path = output_dir / "darpa_binsparse.bsp.h5"

    if binsparse_path.exists():
        print(f"Binsparse file already exists at {binsparse_path}")
        return str(binsparse_path)

    # Download using CURL
    if not tns_path.exists():
        print(f"URL: {url}")
        print(f"Downloading DARPA tensor to {tns_gz_path}")

        try:
            subprocess.run(
                ["curl", "-L", "-o", str(tns_gz_path), url],
                check=True,
            )
            print("Download completed.")
        except FileNotFoundError:
            print("CURL not found. Please install CURL or download the file manually.")
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error downloading file: {e}")
            print("You can manually download from: http://frostt.io/tensors/darpa/")
            print("Look for '1998darpa.tns.gz' and save to:", tns_gz_path)
            return None

        try:
            subprocess.rin(["gunzip", "-f", str(tns_gz_path)], check=True)
            print("Extracted tp: {tns_path}")
        except FileNotFoundError:
            print(
                "gunzip not found. Please install gunzip or extract the file manually."
            )
            return None
        except subprocess.CalledProcessError as e:
            print(f"Error extracting file: {e}")
            return None
    else:
        print(f"Using existing file: {tns_path}")

    # Check if Julia installed
    try:
        result = subprocess.run(
            ["julia", "--version"], capture_output=True, text=True, check=True
        )
        print(f" Found Julia: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Julia not found. Please install Julia to proceed.")
        return None
    except subprocess.CalledProcessError:
        print("Julia command failed.")
        return None

    julia_script = f"""
    try
        using Finch
    catch
        Pkg.add("Finch")
        using Finch
    end

    tensor = Finch.fread("{tns_path}")
    Finch.fwrite("{binsparse_path}", tensor)
    println("Output: {binsparse_path}")
    """
    julia_script_path = output_dir / "convert_darpa_to_binsparse.jl"
    with open(julia_script_path, "w") as f:
        f.write(julia_script)
    print(f"Julia script created at {julia_script_path}")
    try:
        subprocess.run(["julia", str(julia_script_path)], check=True)
        julia_script_path.unlink()
        return str(binsparse_path)
    except subprocess.CalledProcessError as e:
        print(f"Error running Julia script: {e}")
        return None


if __name__ == "__main__":
    print("DARPA Tensor Download and Conversion Script")
    result = download_darpa_tensor()
    if result:
        print(f"DARPA tensor converted to binsparse format at: {result}")
    else:
        print("Failed to download or convert DARPA tensor.")
        sys.exit(1)
