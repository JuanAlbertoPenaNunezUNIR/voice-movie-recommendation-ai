# Script para reconstruir y levantar la aplicación Docker actualizada

Write-Host "🔨 Reconstruyendo aplicación Docker..." -ForegroundColor Cyan

Write-Host ""
Write-Host "Limpiando el entorno actual..."
docker-compose down --rmi all --volumes --remove-orphans

Write-Host ""
Write-Host "Construyendo las nuevas imágenes (sin usar caché para asegurar los cambios)..." -ForegroundColor Yellow
docker-compose build --no-cache

Write-Host ""
Write-Host "Levantando los servicios en segundo plano..." -ForegroundColor Yellow
docker-compose up -d

Write-Host ""
Write-Host "✅ Aplicación reconstruida y en ejecución!" -ForegroundColor Green
Write-Host ""
Write-Host "📊 Ver estado de servicios:" -ForegroundColor Cyan
Write-Host "   docker-compose ps"
Write-Host ""
Write-Host "📋 Ver logs:" -ForegroundColor Cyan
Write-Host "   docker-compose logs -f"
Write-Host ""
Write-Host "🌐 Acceder a:" -ForegroundColor Cyan
Write-Host "   Frontend: http://localhost:8080"
Write-Host "   Backend API: http://localhost:8000"
Write-Host "   Backend Docs: http://localhost:8000/docs"

