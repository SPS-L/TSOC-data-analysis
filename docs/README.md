# TSOC Data Analysis Documentation

This directory contains the comprehensive documentation for TSOC Data Analysis, including detailed license compatibility analysis.

## Documentation Structure

### License Documentation

The license documentation consists of two complementary documents:

1. **`license_summary.rst`** - Quick reference guide for licensing
   - Overview of license compatibility status
   - Key points and requirements
   - Dependency license summary
   - Required actions for compliance

2. **`license.rst`** - Full Apache 2.0 license text
   - Complete license terms and conditions
   - Copyright notice
   - License application instructions

### Building Documentation

To build the documentation:

```bash
cd docs
make html
```

The built documentation will be available in `_build/html/`.

### License Compliance Files

- **`NOTICE`** - Required attribution file for Apache 2.0 compliance
- **`LICENSE`** - Full Apache 2.0 license text

## Key Findings

✅ **All dependencies are fully compatible with Apache 2.0**

The software uses only permissive open-source licenses:
- BSD-3-Clause (pandas, numpy, scikit-learn, scipy, etc.)
- MIT (openpyxl, pytest, black, etc.)
- Custom permissive licenses (matplotlib)

No GPL, LGPL, or other copyleft licenses are used, ensuring maximum flexibility for commercial use and distribution.

## Contact

For questions about licensing or compliance:
- **Sustainable Power Systems Lab (SPSL)**
- **Email**: info@sps-lab.org
- **Website**: https://sps-lab.org 