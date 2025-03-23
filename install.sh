#!/bin/bash

# =================================================================
# INSTALACIÓN COMPLETA DE QuickQuake (SIN COMPILACIÓN DE HYPOINVERSE)
# =================================================================

# 1. Crear ambiente Conda desde environment.yml
echo "Creando ambiente QuickQuake..."
conda env create -f environment.yml || {
    echo "[!] Fallo al crear el ambiente. Verifica environment.yml";
    exit 1;
}

# 2. Activar ambiente (versión corregida)
echo "Activando ambiente..."
source "$(conda info --base)/etc/profile.d/conda.sh"
conda activate QuickQuake || {
    echo "[!] No se pudo activar el ambiente. ¿Está instalado?";
    exit 1;
}

# 3. Instalar paquetes locales en modo editable
echo "Instalando PhaseNet y Gamma..."
pip install -e dependencies/PhaseNet || {
    echo "[!] Fallo al instalar PhaseNet. Verifica la carpeta dependencies/PhaseNet";
    exit 1;
}

pip install -e dependencies/GaMMA || {
    echo "[!] Fallo al instalar Gamma. Verifica la carpeta dependencies/GaMMA";
    exit 1;
}

# 4. Dar permisos a los scripts
echo "Configurando permisos..."
chmod +x quickquake/*.py

# 5. Configurar variables de entorno (hipo ya compilado)
echo "Configurando variables de entorno..."
cat <<EOT >> ~/.bashrc
# QuickQuake configuration
export QUICKQUAKE_ROOT="$(pwd)"
export PATH="$(pwd)/quickquake:\$PATH"
export HYPO_BIN="$(pwd)/dependencies/hyp1.40/source/hyp1.40"
EOT

source ~/.bashrc

# 6. Verificación final
echo ""
echo "================================================================"
echo "¡Instalación completada! Verifica con:"
echo "  \$ conda activate QuickQuake"
echo "  \$ ls -l \$HYPO_BIN  # Debe mostrar el ejecutable (si ya está compilado)"
echo "================================================================"
