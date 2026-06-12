"""
One-shot patcher: run once from the root of your TranAD fork.

  python patch_tranad.py

What it does:
  1. Adds __init__.py so the repo becomes a Python package
  2. Rewrites `from src.` -> `from tranad.src.` in all files
  3. Fixes PyTorch 2.x incompatibilities:
     - adds **kwargs to custom Transformer layer forward() (absorbs is_causal)
     - sets enable_nested_tensor=False on TransformerEncoder
  4. Makes dgl import optional (only needed for the GDN model)
"""

import os
import re

# Script sits in the fork root, so the repo root IS the TranAD directory
TRANAD = os.path.dirname(os.path.abspath(__file__))


def patch_text(path, replacements):
    if not os.path.exists(path):
        print(f"  skipped (missing): {path}")
        return
    with open(path) as f:
        text = f.read()
    new = text
    for pattern, repl in replacements:
        new = re.sub(pattern, repl, new)
    if new != text:
        with open(path, "w") as f:
            f.write(new)
        print(f"  patched: {os.path.relpath(path, TRANAD)}")
    else:
        print(f"  unchanged: {os.path.relpath(path, TRANAD)}")


def main():
    print(f"Patching TranAD at {TRANAD}")

    # 1. Make it a package
    for d in [TRANAD, os.path.join(TRANAD, "src")]:
        init = os.path.join(d, "__init__.py")
        if not os.path.exists(init):
            open(init, "w").close()
            print(f"  created: {os.path.relpath(init, TRANAD)}")

    # 2. Rewrite imports in every .py file
    import_fix = [
        (r"\bfrom src\.", "from tranad.src."),
        (r"\bimport src\.", "import tranad.src."),
        (r"\bimport src\b", "import tranad.src as src"),
    ]
    for root, _, files in os.walk(TRANAD):
        if ".git" in root:
            continue
        for fn in files:
            if fn.endswith(".py") and fn not in ("setup.py", "__init__.py", "patch_tranad.py"):
                patch_text(os.path.join(root, fn), import_fix)

    # 3. PyTorch 2.x fixes
    patch_text(
        os.path.join(TRANAD, "src", "dlutils.py"),
        [
            (
                r"def forward\(self, src,\s*src_mask=None, src_key_padding_mask=None\):",
                "def forward(self, src, src_mask=None, src_key_padding_mask=None, **kwargs):",
            ),
            (
                r"def forward\(self, tgt, memory, tgt_mask=None, memory_mask=None, "
                r"tgt_key_padding_mask=None, memory_key_padding_mask=None\):",
                "def forward(self, tgt, memory, tgt_mask=None, memory_mask=None, "
                "tgt_key_padding_mask=None, memory_key_padding_mask=None, **kwargs):",
            ),
        ],
    )
    patch_text(
        os.path.join(TRANAD, "src", "models.py"),
        [
            (
                r"TransformerEncoder\(encoder_layers, 1\)",
                "TransformerEncoder(encoder_layers, 1, enable_nested_tensor=False)",
            ),
            # dgl is only needed for the GDN model -> make it optional
            (
                r"import dgl\nfrom dgl\.nn import GATConv",
                "try:\n"
                "    import dgl\n"
                "    from dgl.nn import GATConv\n"
                "except ImportError:  # dgl only needed for GDN model\n"
                "    dgl, GATConv = None, None",
            ),
        ],
    )

    print("Done. Now commit, push, then: pip install git+https://github.com/<you>/TranAD.git")


if __name__ == "__main__":
    main()
