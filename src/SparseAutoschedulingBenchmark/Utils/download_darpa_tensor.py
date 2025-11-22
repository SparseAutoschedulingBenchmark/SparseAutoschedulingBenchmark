"""
Script to download and convert DARPA tensor from FROSTT to binsparse format.
The DARPA tensor represents intrusion detection data with 28,436,033 non-zeros.
This is the smallest 3D tensor from FROSTT (exclusing matrix multiplication).

Prerequisites:
- Julia must be installed: https://julialang.org/downloads/
- Finch.jl package must be installed: julia -e 'using Pkg; Pkg.add("TensorMarket")'
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

    url = "https://frostt-tensors.s3.us-east-2.amazonaws.com/1998DARPA/1998darpa.tns.gz"
    tns_gz_path = output_dir / "1998darpa.tns.gz"
    tns_path = output_dir / "1998darpa.tns"
    binsparse_path = output_dir / "darpa_tensor.bsp.h5"

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
            import gzip
            import shutil
            import tarfile

            if tarfile.is_tarfile(tns_gz_path):
                print("File is a tar archive, extracting...")
                with tarfile.open(tns_gz_path, "r:gz") as tar:
                    members = tar.getmembers()
                    print(f"Archive contains {len(members)} files")
                    # Finding the .tns file
                    tns_member = None
                    for member in members:
                        if (
                            not member.name.startswith("._")
                            and not member.name.startswith("PaxHeader")
                            and member.name.endswith(".tns")
                            and member.isfile()
                        ):
                            tns_member = member
                            print(f"Found tensor file: {member.name}")
                            break
                    if tns_member is None:
                        print("ERROR: Could not find .tns file in archive")
                        print("Archive contents:")
                        for m in members:
                            print(f"    - {m.name}")
                        return None
                    print(f"Extracting {tns_member.name}...")
                    tar.extract(tns_member, output_dir)
                    extracted_path = output_dir / tns_member.name
                    if extracted_path != tns_path:
                        shutil.move(str(extracted_path), str(tns_path))
                    print(f"Extracted to: {tns_path}")
            else:
                print("File is a regular gzip, extracting...")
                with (
                    gzip.open(tns_gz_path, "rb") as f_in,
                    open(tns_path, "wb") as f_out,
                ):
                    shutil.copyfileobj(f_in, f_out)
                print(f"Extracted to: {tns_path}")
            print("\nVerifying extracted file...")
            with open(tns_path, encoding="utf-8", errors="ignore") as f:
                first_line = f.readline().strip()
                if first_line and all(
                    c.isdigit() or c.isspace() or c == "." or c == "-"
                    for c in first_line[:50]
                ):
                    print("File appears to be valid .tns format")
                else:
                    print("Warning: File may not be in .tns format")
            tns_gz_path.unlink()
            print("Cleaned up archive file")
        except (tarfile.TarError, gzip.BadGzipFile, OSError) as e:
            print(f"Error extracting file: {e}")
            return None
    else:
        print(f"Using existing file: {tns_path}")

    # Check if Julia installed
    try:
        result = subprocess.run(
            ["julia", "--version"], capture_output=True, text=True, check=True
        )
        print(f"Found Julia: {result.stdout.strip()}")
    except FileNotFoundError:
        print("Julia not found. Please install Julia to proceed.")
        return None
    except subprocess.CalledProcessError:
        print("Julia command failed.")
        return None

    tns_path_julia = str(tns_path).replace("\\", "/")
    binsparse_path_julia = str(binsparse_path).replace("\\", "/")

    julia_script = f"""
    using Pkg
    println("Checking for TensorMarket.jl package...")
    try
        using TensorMarket
    catch
        Pkg.add("TensorMarket")
        using TensorMarket
    end

    try
        using HDF5
    catch
        Pkg.add("HDF5")
        using HDF5
    end

    tensor = TensorMarket.tnsread("{tns_path_julia}")
    println("Extracting COO format data...")
    coords, values = tensor
    indices_0 = coords[1] .- 1
    indices_1 = coords[2] .- 1
    indices_2 = coords[3] .- 1
    shape = (maximum(indices_0) + 1,
             maximum(indices_1) + 1,
             maximum(indices_2) + 1)

    h5open("{binsparse_path_julia}", "w") do file
        file["indices_0"] = Int32.(indices_0)
        file["indices_1"] = Int32.(indices_1)
        file["indices_2"] = Int32.(indices_2)
        file["values"] = Float32.(values)
        attributes(file)["shape"] = collect(Int64.(shape))
    end
    println("\\nConversion complete!")
    println("Output: {binsparse_path_julia}")
    println("Success: DARPA tensor ready for benchmarking!")
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
