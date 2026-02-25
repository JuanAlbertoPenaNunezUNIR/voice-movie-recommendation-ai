# 🐳 Comandos Docker - Limpieza y Reconstrucción

## 🧹 Limpiar Docker y Liberar Espacio

### Windows (PowerShell):
```powershell
.\limpiar_docker.ps1
```

### Linux/Mac:
```bash
chmod +x limpiar_docker.sh
./limpiar_docker.sh
```

### O manualmente (comandos individuales):

```bash
# 1. Detener y eliminar contenedores
docker-compose down -v

# 2. Eliminar imágenes específicas del proyecto
docker images | grep "movie_" | awk '{print $3}' | xargs docker rmi -f

# 3. Limpiar caché y builds antiguos
docker system prune -a -f --volumes

# Ver espacio liberado
docker system df
```

## 🔨 Reconstruir y Levantar Aplicación Actualizada

### Windows (PowerShell):
```powershell
.\rebuild_and_start.ps1
```

### Linux/Mac:
```bash
chmod +x rebuild_and_start.sh
./rebuild_and_start.sh
```

### O manualmente:

```bash
# Reconstruir sin caché (asegura usar código actualizado)
docker-compose build --no-cache

# Levantar servicios
docker-compose up -d

# Ver logs
docker-compose logs -f
```

## 📋 Comandos Útiles

```bash
# Ver estado de servicios
docker-compose ps

# Ver uso de recursos
docker stats

# Ver logs de un servicio específico
docker-compose logs -f backend
docker-compose logs -f frontend

# Reiniciar un servicio
docker-compose restart backend

# Parar servicios (sin eliminar)
docker-compose stop

# Parar y eliminar contenedores
docker-compose down
```

## 💾 Verificar Espacio en Docker

```bash
# Ver espacio usado por Docker
docker system df

# Ver detalles de volúmenes
docker volume ls
docker volume inspect <nombre_volumen>

# Ver imágenes
docker images
```

