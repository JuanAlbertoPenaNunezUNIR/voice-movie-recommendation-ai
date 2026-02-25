# Script para limpiar completamente Docker y liberar espacio
# Elimina contenedores, imágenes, volúmenes no usados y caché

Write-Host "🧹 Limpiando Docker completamente..." -ForegroundColor Cyan
Write-Host ""

# 1. Detener y eliminar todos los contenedores del proyecto
Write-Host "1️⃣ Deteniendo y eliminando contenedores..." -ForegroundColor Yellow
docker-compose down -v 2>$null

# 2. Eliminar imágenes específicas del proyecto (si existen)
Write-Host "2️⃣ Eliminando imágenes del proyecto..." -ForegroundColor Yellow
docker images | Select-String "movie_" | ForEach-Object {
    $imageId = ($_ -split '\s+')[2]
    if ($imageId -and $imageId -ne "IMAGE") {
        docker rmi -f $imageId 2>$null
    }
}

# 3. Eliminar contenedores huérfanos
Write-Host "3️⃣ Eliminando contenedores huérfanos..." -ForegroundColor Yellow
docker container prune -f

# 4. Eliminar imágenes no utilizadas
Write-Host "4️⃣ Eliminando imágenes no utilizadas..." -ForegroundColor Yellow
docker image prune -a -f

# 5. Eliminar volúmenes no utilizados (¡CUIDADO! Puede eliminar datos persistentes)
Write-Host "5️⃣ Eliminando volúmenes no utilizados..." -ForegroundColor Yellow
docker volume prune -f

# 6. Limpiar caché de construcción
Write-Host "6️⃣ Limpiando caché de construcción..." -ForegroundColor Yellow
docker builder prune -a -f

# 7. Limpiar todo el sistema (opcional, más agresivo)
Write-Host "7️⃣ Limpieza completa del sistema..." -ForegroundColor Yellow
docker system prune -a -f --volumes

Write-Host ""
Write-Host "✅ Limpieza completada!" -ForegroundColor Green
Write-Host ""
Write-Host "💾 Para ver el espacio liberado:" -ForegroundColor Cyan
Write-Host "   docker system df"

