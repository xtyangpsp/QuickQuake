#!/bin/bash

# --------------------------------------------------
# Configuración Automática de QuickQuake
# --------------------------------------------------

# Ruta absoluta al directorio raíz de QuickQuake (ajusta esto!)
QUICKQUAKE_ROOT="/home/elizabeth/soft/my_scripts/My_quakeFlow_1"
PHASENET_ROOT="/home/elizabeth/soft/src/QuakeFlow/PhaseNet"

# 1. Agregar rutas críticas al PATH y PYTHONPATH
echo "export QUICKQUAKE_ROOT=\"$QUICKQUAKE_ROOT\"" >> ~/.bash_profile
echo "export PHASENET_ROOT=\"$PHASENET_ROOT\"" >> ~/.bash_profile
echo "export PATH=\"$QUICKQUAKE_ROOT:\$PATH\"" >> ~/.bash_profile
echo "export PYTHONPATH=\"$QUICKQUAKE_ROOT:$PHASENET_ROOT:\$PYTHONPATH\"" >> ~/.bash_profile

# 2. Hacer scripts ejecutables
find "$QUICKQUAKE_ROOT" -name "QQ_*.py" -exec chmod +x {} \;

# 3. Mensaje final
echo "¡Configuración completada! Ejecuta:"
echo "source ~/.bash_profile"
