#!/bin/bash
# Script de inicio para Render

# Crear directorios necesarios
mkdir -p uploads graphs

# Iniciar la aplicación con gunicorn
gunicorn --bind 0.0.0.0:$PORT app:app