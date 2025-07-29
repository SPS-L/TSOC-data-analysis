#!/bin/bash

# Build Documentation Script for TSOC Data Analysis
# This script builds both HTML and PDF documentation

set -e  # Exit on any error

echo "Building TSOC Data Analysis Documentation..."

# Build HTML documentation
echo "Building HTML documentation..."
make html
echo "HTML documentation built successfully!"

# Build PDF documentation
echo "Building PDF documentation..."
make pdf
echo "PDF documentation built successfully!"

echo ""
echo "Documentation build completed!"
echo "HTML files: _build/html/"
echo "PDF file: _build/latex/TSOCDataAnalysis.pdf"
echo ""
echo "To view HTML documentation, open _build/html/index.html in your browser"
echo "To view PDF documentation, open _build/latex/TSOCDataAnalysis.pdf" 