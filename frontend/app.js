// API Configuration
const API_BASE = 'http://localhost:8001';

// State
let map = null;
let markers = [];
let currentData = null;

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    initMap();
    setupEventListeners();
    loadHistoricalData();
});

// Initialize Leaflet Map with dark theme
function initMap() {
    map = L.map('map', {
        center: [40.4168, -3.7038],
        zoom: 11,
        zoomControl: true,
    });
    
    // Dark theme tile layer
    L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
        attribution: '© OpenStreetMap contributors © CARTO',
        maxZoom: 19,
    }).addTo(map);
    
    // Add other layers
    const satelliteLayer = L.tileLayer('https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}', {
        attribution: 'Esri',
        maxZoom: 19,
    });
    
    const osmLayer = L.tileLayer('https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png', {
        attribution: '© OpenStreetMap contributors',
        maxZoom: 19,
    });
    
    const baseMaps = {
        "Modo Oscuro": L.tileLayer('https://{s}.basemaps.cartocdn.com/dark_all/{z}/{x}/{y}{r}.png', {
            attribution: '© OpenStreetMap contributors © CARTO',
            maxZoom: 19,
        }),
        "Satélite": satelliteLayer,
        "OpenStreetMap": osmLayer,
    };
    
    L.control.layers(baseMaps).addTo(map);
}


// Setup Event Listeners
function setupEventListeners() {
    document.getElementById('btn-update').addEventListener('click', updateRealtimeData);
    document.getElementById('btn-predict').addEventListener('click', runPredictions);
    document.getElementById('btn-center').addEventListener('click', () => {
        map.setView([40.4168, -3.7038], 11);
        showToast('Mapa centrado', 'success');
    });
    document.getElementById('btn-fit').addEventListener('click', fitMapBounds);
}

// Load Historical Data
async function loadHistoricalData() {
    showLoading();
    try {
        console.log('Loading historical data from:', `${API_BASE}/api/data/historical`);
        const response = await fetch(`${API_BASE}/api/data/historical`, {
            signal: AbortSignal.timeout(15000) // 15 second timeout
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            throw new Error(`HTTP error! status: ${response.status}`);
        }
        
        const data = await response.json();
        console.log('Historical data received:', data);
        
        if (data.stations && data.stations.length > 0) {
            console.log('Processing', data.stations.length, 'stations');
            currentData = data;
            try {
                updateAll(data);
                console.log('Data updated successfully');
            } catch (error) {
                console.error('Error updating UI:', error);
            }
            hideLoading();
        } else {
            // Si no hay datos históricos, intentar cargar datos en tiempo real
            console.log('No hay datos históricos, cargando datos en tiempo real...');
            hideLoading();
            await updateRealtimeData();
        }
    } catch (error) {
        console.error('Error loading historical data:', error);
        hideLoading();
        showToast('Error cargando datos históricos. Intentando datos en tiempo real...', 'error');
        // Intentar cargar datos en tiempo real como fallback
        try {
            await updateRealtimeData();
        } catch (e) {
            console.error('Error loading realtime data:', e);
            showToast('Error cargando datos', 'error');
        }
    }
}

// Update Realtime Data
async function updateRealtimeData() {
    const btn = document.getElementById('btn-update');
    if (btn) btn.disabled = true;
    showLoading();
    
    try {
        console.log('Loading realtime data from:', `${API_BASE}/api/data/realtime`);
        const response = await fetch(`${API_BASE}/api/data/realtime`, {
            signal: AbortSignal.timeout(30000) // 30 second timeout
        });
        
        console.log('Response status:', response.status);
        
        if (!response.ok) {
            const errorText = await response.text();
            throw new Error(`HTTP error! status: ${response.status}, body: ${errorText}`);
        }
        
        const data = await response.json();
        console.log('Realtime data received:', data);
        
        if (data.stations && data.stations.length > 0) {
            currentData = data;
            updateAll(data);
            if (btn) showToast('✅ Datos actualizados', 'success');
        } else {
            if (btn) showToast('❌ No se pudieron obtener datos', 'error');
        }
    } catch (error) {
        console.error('Error updating data:', error);
        if (btn) showToast(`Error actualizando datos: ${error.message}`, 'error');
    } finally {
        if (btn) btn.disabled = false;
        hideLoading();
    }
}

// Run Predictions
async function runPredictions() {
    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    showLoading();
    
    try {
        const useRealtime = !!(currentData && currentData.timestamp);
        const response = await fetch(`${API_BASE}/api/predict`, {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ use_realtime: useRealtime }),
        });
        
        const data = await response.json();
        
        if (data.predictions && data.predictions.length > 0) {
            updatePredictions(data);
            updateMapWithPredictions(data.predictions);
            updateTopStations(data.predictions);
            showToast(`✅ ${data.good_count} lugares buenos para correr`, 'success');
        } else {
            showToast('No se pudieron generar predicciones', 'error');
        }
    } catch (error) {
        console.error('Error running predictions:', error);
        showToast('Error ejecutando predicciones', 'error');
    } finally {
        btn.disabled = false;
        hideLoading();
    }
}

// Update All
function updateAll(data) {
    try {
        console.log('Updating all components...');
        if (data.stations) {
            updateHeroStats(data);
        }
        if (data.stations) {
            updateMetrics(data);
        }
        if (data.weather) {
            updateWeather(data.weather);
        }
        if (data.stations && data.stations.length > 0) {
            updateMap(data.stations);
        }
        console.log('All components updated');
    } catch (error) {
        console.error('Error in updateAll:', error);
        throw error;
    }
}

// Update Hero Stats
function updateHeroStats(data) {
    if (!data.stations) return;
    
    const stations = data.stations;
    const avgAqi = stations.reduce((sum, s) => sum + (s.aqi || 0), 0) / stations.length;
    const goodCount = stations.filter(s => s.is_good_to_run).length || 0;
    
    animateValue('hero-stations', 0, stations.length, 800, 0);
    animateValue('hero-aqi', 0, avgAqi, 1000, 1);
    animateValue('hero-good', 0, goodCount, 1000, 0);
}

// Update Metrics
function updateMetrics(data) {
    if (!data.stations || data.stations.length === 0) return;
    
    try {
        const stations = data.stations;
        const avgAqi = stations.reduce((sum, s) => sum + (s.aqi || 0), 0) / stations.length;
        
        const aqiEl = document.getElementById('metric-aqi');
        if (aqiEl) animateValue('metric-aqi', 0, avgAqi, 1000, 1);
        
        const aqiBadge = document.getElementById('aqi-badge');
        if (aqiBadge) {
            aqiBadge.textContent = getAqiLabel(avgAqi);
            aqiBadge.className = 'card-badge ' + getAqiBadgeClass(avgAqi);
        }
        
        // Average pollutants
        const no2Stations = stations.filter(s => s.no2 !== null && s.no2 !== undefined);
        const o3Stations = stations.filter(s => s.o3 !== null && s.o3 !== undefined);
        const pm10Stations = stations.filter(s => s.pm10 !== null && s.pm10 !== undefined);
        const pm25Stations = stations.filter(s => s.pm25 !== null && s.pm25 !== undefined);
        
        const avgNo2 = no2Stations.length > 0 ? no2Stations.reduce((sum, s) => sum + s.no2, 0) / no2Stations.length : 0;
        const avgO3 = o3Stations.length > 0 ? o3Stations.reduce((sum, s) => sum + s.o3, 0) / o3Stations.length : 0;
        const avgPm10 = pm10Stations.length > 0 ? pm10Stations.reduce((sum, s) => sum + s.pm10, 0) / pm10Stations.length : 0;
        const avgPm25 = pm25Stations.length > 0 ? pm25Stations.reduce((sum, s) => sum + s.pm25, 0) / pm25Stations.length : 0;
        
        const no2El = document.getElementById('metric-no2');
        const o3El = document.getElementById('metric-o3');
        const pm10El = document.getElementById('metric-pm10');
        const pm25El = document.getElementById('metric-pm25');
        
        if (no2El) no2El.textContent = avgNo2 > 0 ? avgNo2.toFixed(1) : '--';
        if (o3El) o3El.textContent = avgO3 > 0 ? avgO3.toFixed(1) : '--';
        if (pm10El) pm10El.textContent = avgPm10 > 0 ? avgPm10.toFixed(1) : '--';
        if (pm25El) pm25El.textContent = avgPm25 > 0 ? avgPm25.toFixed(1) : '--';
    } catch (error) {
        console.error('Error updating metrics:', error);
    }
}

// Update Weather
function updateWeather(weather) {
    if (!weather) {
        console.log('No weather data available');
        return;
    }
    
    console.log('Updating weather with data:', weather);
    
    const card = document.getElementById('weather-card');
    if (card) {
        card.style.display = 'block';
    }
    
    // Update icon based on weather
    const icon = document.getElementById('weather-icon');
    if (icon && weather.weather_code !== null && weather.weather_code !== undefined) {
        if (weather.weather_code >= 61 && weather.weather_code <= 82) {
            icon.textContent = '🌧️';
        } else if (weather.weather_code >= 71 && weather.weather_code <= 75) {
            icon.textContent = '❄️';
        } else if (weather.weather_code >= 95 && weather.weather_code <= 99) {
            icon.textContent = '⛈️';
        } else if (weather.weather_code >= 45 && weather.weather_code <= 48) {
            icon.textContent = '🌫️';
        } else if (weather.weather_code >= 2 && weather.weather_code <= 3) {
            icon.textContent = '☁️';
        } else {
            icon.textContent = '☀️';
        }
    }
    
    // Update temperature
    const tempEl = document.getElementById('weather-temp');
    if (tempEl && weather.temperature !== null && weather.temperature !== undefined) {
        console.log('Setting temperature:', weather.temperature);
        const tempValue = parseFloat(weather.temperature);
        if (!isNaN(tempValue)) {
            tempEl.textContent = ''; // Clear first
            animateValue('weather-temp', 0, tempValue, 800, 1);
            // The animateValue will set the value, but we need to add °C after
            setTimeout(() => {
                const currentText = tempEl.textContent;
                if (!currentText.includes('°C')) {
                    tempEl.textContent = tempValue.toFixed(1) + '°C';
                }
            }, 850);
        }
    } else {
        console.log('Temperature not available:', weather.temperature);
    }
    
    // Update humidity
    const humidityEl = document.getElementById('weather-humidity');
    if (humidityEl && weather.humidity !== null && weather.humidity !== undefined) {
        const humidityValue = parseFloat(weather.humidity);
        if (!isNaN(humidityValue)) {
            humidityEl.textContent = humidityValue.toFixed(0) + '%';
        }
    }
    
    // Update wind speed
    const windEl = document.getElementById('weather-wind');
    if (windEl && weather.wind_speed !== null && weather.wind_speed !== undefined) {
        const windValue = parseFloat(weather.wind_speed);
        if (!isNaN(windValue)) {
            windEl.textContent = windValue.toFixed(1) + ' km/h';
        }
    }
    
    // Update description
    const descEl = document.getElementById('weather-desc');
    if (descEl && weather.description) {
        descEl.textContent = weather.description;
    }
    
    // Show forecast indicator
    if (weather.is_forecast) {
        const cardHeader = card ? card.querySelector('.card-header') : null;
        if (cardHeader && !cardHeader.querySelector('.forecast-badge')) {
            const badge = document.createElement('div');
            badge.className = 'forecast-badge';
            badge.textContent = '📅 Predicción 1h';
            badge.style.cssText = 'font-size: 0.75rem; color: #8b5cf6; background: rgba(139, 92, 246, 0.2); padding: 0.25rem 0.5rem; border-radius: 6px;';
            cardHeader.appendChild(badge);
        }
    }
}

// Update Predictions
function updatePredictions(data) {
    const card = document.getElementById('predictions-card');
    card.style.display = 'block';
    
    document.getElementById('prediction-status').textContent = 'Ejecutado';
    document.getElementById('prediction-status').style.background = 'var(--success)';
    document.getElementById('prediction-status').style.color = 'white';
    
    animateValue('pred-good-count', 0, data.good_count, 800, 0);
    animateValue('pred-bad-count', 0, data.total - data.good_count, 800, 0);
}

// Update Map
function updateMap(stations) {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    stations.forEach((station, index) => {
        if (!station.latitude || !station.longitude) return;
        
        const aqi = station.aqi || 0;
        const color = getAqiColor(aqi);
        
        const marker = L.circleMarker([station.latitude, station.longitude], {
            radius: station.temperature ? Math.max(8, Math.min(20, station.temperature / 2)) : 10,
            fillColor: color,
            color: color,
            weight: 2,
            opacity: 1,
            fillOpacity: 0.7,
        });
        
        const popup = createPopup(station, color);
        marker.bindPopup(popup, { maxWidth: 300, className: 'custom-popup' });
        
        setTimeout(() => {
            marker.addTo(map);
            markers.push(marker);
        }, index * 30);
    });
}

// Update Map with Predictions
function updateMapWithPredictions(predictions) {
    markers.forEach(marker => map.removeLayer(marker));
    markers = [];
    
    predictions.forEach((station, index) => {
        if (!station.latitude || !station.longitude) return;
        
        const aqi = station.aqi || 0;
        const color = getAqiColor(aqi);
        
        const marker = L.circleMarker([station.latitude, station.longitude], {
            radius: station.temperature ? Math.max(8, Math.min(20, station.temperature / 2)) : 10,
            fillColor: color,
            color: color,
            weight: station.is_good_to_run ? 4 : 2,
            opacity: 1,
            fillOpacity: station.is_good_to_run ? 0.9 : 0.7,
        });
        
        const popup = createPopupWithPrediction(station, color);
        marker.bindPopup(popup, { maxWidth: 300, className: 'custom-popup' });
        
        setTimeout(() => {
            marker.addTo(map);
            markers.push(marker);
        }, index * 30);
    });
}

// Create Popup
function createPopup(station, color) {
    return `
        <div style="font-family: 'Inter', sans-serif; color: #ffffff;">
            <h3 style="margin: 0 0 0.5rem 0; font-size: 1.1rem; font-weight: 700;">${station.station_name}</h3>
            <p style="margin: 0.25rem 0; color: #a1a1aa; font-size: 0.85rem;">${station.station_type}</p>
            <div style="background: ${color}20; padding: 0.75rem; border-radius: 8px; margin: 0.75rem 0; border-left: 3px solid ${color};">
                <div style="color: ${color}; font-weight: 700; margin-bottom: 0.25rem;">AQI: ${station.aqi.toFixed(1)}</div>
                <div style="font-size: 0.85rem; color: #a1a1aa;">
                    NO₂: ${station.no2 !== null ? station.no2.toFixed(1) : '--'} | 
                    O₃: ${station.o3 !== null ? station.o3.toFixed(1) : '--'}
                </div>
            </div>
            <div style="margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1);">
                <div style="font-size: 1.2rem; font-weight: 700;">${station.temperature !== null ? station.temperature.toFixed(1) : '--'}°C</div>
                <div style="font-size: 0.85rem; color: #a1a1aa; margin-top: 0.25rem;">
                    💨 ${station.wind_speed !== null ? station.wind_speed.toFixed(1) : '--'} km/h
                </div>
            </div>
        </div>
    `;
}

// Create Popup with Prediction
function createPopupWithPrediction(station, color) {
    const prediction = station.is_good_to_run 
        ? '<div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(16, 185, 129, 0.2); border-radius: 6px; text-align: center; font-weight: 600; color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3);">✅ Bueno para correr</div>'
        : '<div style="margin-top: 0.75rem; padding: 0.5rem; background: rgba(239, 68, 68, 0.2); border-radius: 6px; text-align: center; font-weight: 600; color: #ef4444; border: 1px solid rgba(239, 68, 68, 0.3);">❌ No recomendado</div>';
    
    return createPopup(station, color) + prediction;
}


// Update Top Stations
function updateTopStations(predictions) {
    const list = document.getElementById('stations-list');
    list.innerHTML = '';
    
    predictions.slice(0, 5).forEach((station, index) => {
        const item = document.createElement('div');
        item.className = 'station-item';
        item.style.animationDelay = `${index * 0.1}s`;
        item.style.opacity = '0';
        item.style.animation = 'fadeInUp 0.5s ease forwards';
        
        const explanation = station.explanation || {};
        const reasons = explanation.reasons || [];
        const warnings = explanation.warnings || [];
        const summary = explanation.summary || '';
        
        // Crear HTML de explicación
        let explanationHTML = '';
        if (reasons.length > 0 || warnings.length > 0 || summary) {
            explanationHTML = `
                <div class="station-explanation" style="display: none; margin-top: 0.75rem; padding-top: 0.75rem; border-top: 1px solid rgba(255,255,255,0.1);">
                    ${summary ? `<div style="color: #a1a1aa; font-size: 0.85rem; margin-bottom: 0.5rem; font-style: italic;">${summary}</div>` : ''}
                    ${reasons.length > 0 ? `
                        <div style="margin-bottom: 0.5rem;">
                            <div style="color: #10b981; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.25rem;">✅ Ventajas:</div>
                            <ul style="margin: 0; padding-left: 1.25rem; color: #d1d5db; font-size: 0.8rem; line-height: 1.5;">
                                ${reasons.map(r => `<li>${r}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                    ${warnings.length > 0 ? `
                        <div>
                            <div style="color: #ef4444; font-size: 0.75rem; font-weight: 600; margin-bottom: 0.25rem;">⚠️ Consideraciones:</div>
                            <ul style="margin: 0; padding-left: 1.25rem; color: #fca5a5; font-size: 0.8rem; line-height: 1.5;">
                                ${warnings.map(w => `<li>${w}</li>`).join('')}
                            </ul>
                        </div>
                    ` : ''}
                </div>
            `;
        }
        
        item.innerHTML = `
            <div class="station-info">
                <div class="station-name">${station.station_name}</div>
                <div class="station-meta">${station.station_type}</div>
            </div>
            <div class="station-stats">
                <div class="station-stat">
                    <div class="station-stat-value" style="color: ${getScoreColor(station.running_score || 0)}; font-weight: 700;">
                        ${Math.round(station.running_score || 0)}
                    </div>
                    <div class="station-stat-label">Score</div>
                </div>
                <div class="station-stat">
                    <div class="station-stat-value">${station.aqi.toFixed(1)}</div>
                    <div class="station-stat-label">AQI</div>
                </div>
                <span class="station-badge ${station.is_good_to_run ? 'badge-success' : 'badge-danger'}">
                    ${station.is_good_to_run ? '✅' : '❌'}
                </span>
            </div>
            ${explanationHTML}
            <div style="margin-top: 0.5rem; text-align: center;">
                <button class="explanation-toggle" style="background: transparent; border: 1px solid rgba(255,255,255,0.2); color: #a1a1aa; padding: 0.25rem 0.75rem; border-radius: 4px; font-size: 0.75rem; cursor: pointer; transition: all 0.2s;">
                    ${explanationHTML ? '📊 Ver detalles' : ''}
                </button>
            </div>
        `;
        
        // Toggle explicación
        const toggleBtn = item.querySelector('.explanation-toggle');
        const explanationDiv = item.querySelector('.station-explanation');
        if (toggleBtn && explanationDiv) {
            toggleBtn.addEventListener('click', (e) => {
                e.stopPropagation();
                const isVisible = explanationDiv.style.display !== 'none';
                explanationDiv.style.display = isVisible ? 'none' : 'block';
                toggleBtn.textContent = isVisible ? '📊 Ver detalles' : '📊 Ocultar detalles';
            });
        }
        
        // Click en el item para centrar en el mapa
        item.addEventListener('click', (e) => {
            if (e.target !== toggleBtn && !toggleBtn.contains(e.target)) {
                map.setView([station.latitude, station.longitude], 14);
                showToast(`Centrado en ${station.station_name}`, 'success');
            }
        });
        
        list.appendChild(item);
    });
}

// Fit Map Bounds
function fitMapBounds() {
    if (markers.length === 0) return;
    const group = new L.featureGroup(markers);
    map.fitBounds(group.getBounds().pad(0.1));
    showToast('Vista ajustada', 'success');
}

// Animate Value
function animateValue(id, start, end, duration, decimals) {
    const element = document.getElementById(id);
    if (!element) return;
    
    const range = end - start;
    const increment = range / (duration / 16);
    let current = start;
    
    const timer = setInterval(() => {
        current += increment;
        if ((increment > 0 && current >= end) || (increment < 0 && current <= end)) {
            element.textContent = end.toFixed(decimals);
            clearInterval(timer);
        } else {
            element.textContent = current.toFixed(decimals);
        }
    }, 16);
}

// Helper Functions
function getScoreColor(score) {
    // Color basado en el score: verde (alto) -> amarillo (medio) -> rojo (bajo)
    if (score >= 80) return '#10b981'; // Verde - Excelente
    if (score >= 60) return '#84cc16'; // Verde claro - Bueno
    if (score >= 40) return '#eab308'; // Amarillo - Moderado
    if (score >= 20) return '#f59e0b'; // Naranja - Regular
    return '#ef4444'; // Rojo - Malo
}

function getAqiColor(aqi) {
    if (aqi <= 25) return '#10b981';
    if (aqi <= 50) return '#fbbf24';
    if (aqi <= 75) return '#f97316';
    return '#ef4444';
}

function getAqiLabel(aqi) {
    if (aqi <= 25) return 'Excelente';
    if (aqi <= 50) return 'Buena';
    if (aqi <= 75) return 'Moderada';
    return 'Mala';
}

function getAqiBadgeClass(aqi) {
    if (aqi <= 25) return '';
    if (aqi <= 50) return '';
    if (aqi <= 75) return '';
    return '';
}

// Toast
function showToast(message, type = 'success') {
    const container = document.getElementById('toast-container');
    const toast = document.createElement('div');
    toast.className = `toast ${type}`;
    toast.textContent = message;
    
    container.appendChild(toast);
    
    setTimeout(() => {
        toast.style.animation = 'slideOutRight 0.3s ease-out forwards';
        setTimeout(() => container.removeChild(toast), 300);
    }, 3000);
}

// Loading
function showLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.remove('hidden');
        overlay.style.display = 'flex';
    }
}

function hideLoading() {
    const overlay = document.getElementById('loading-overlay');
    if (overlay) {
        overlay.classList.add('hidden');
        overlay.style.display = 'none';
    }
    console.log('Loading hidden');
}

// Add CSS animations
const style = document.createElement('style');
style.textContent = `
    @keyframes slideOutRight {
        to { transform: translateX(400px); opacity: 0; }
    }
    .custom-popup .leaflet-popup-content-wrapper {
        background: #1a1a24 !important;
        color: #ffffff !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.1) !important;
    }
    .custom-popup .leaflet-popup-tip {
        background: #1a1a24 !important;
    }
    .badge-success {
        background: rgba(16, 185, 129, 0.2) !important;
        color: #10b981 !important;
        border: 1px solid rgba(16, 185, 129, 0.3) !important;
    }
    .badge-danger {
        background: rgba(239, 68, 68, 0.2) !important;
        color: #ef4444 !important;
        border: 1px solid rgba(239, 68, 68, 0.3) !important;
    }
`;
document.head.appendChild(style);
