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
        const response = await fetch(`${API_BASE}/api/data/historical`);
        const data = await response.json();
        
        if (data.stations && data.stations.length > 0) {
            currentData = data;
            updateAll(data);
        }
    } catch (error) {
        console.error('Error loading historical data:', error);
        showToast('Error cargando datos históricos', 'error');
    } finally {
        hideLoading();
    }
}

// Update Realtime Data
async function updateRealtimeData() {
    const btn = document.getElementById('btn-update');
    btn.disabled = true;
    showLoading();
    
    try {
        const response = await fetch(`${API_BASE}/api/data/realtime`);
        const data = await response.json();
        
        if (data.stations && data.stations.length > 0) {
            currentData = data;
            updateAll(data);
            showToast('✅ Datos actualizados', 'success');
        } else {
            showToast('❌ No se pudieron obtener datos', 'error');
        }
    } catch (error) {
        console.error('Error updating data:', error);
        showToast('Error actualizando datos', 'error');
    } finally {
        btn.disabled = false;
        hideLoading();
    }
}

// Run Predictions
async function runPredictions() {
    const btn = document.getElementById('btn-predict');
    btn.disabled = true;
    showLoading();
    
    try {
        const useRealtime = currentData && currentData.timestamp;
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
    updateHeroStats(data);
    updateMetrics(data);
    updateWeather(data.weather);
    updateMap(data.stations);
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
    
    const stations = data.stations;
    const avgAqi = stations.reduce((sum, s) => sum + (s.aqi || 0), 0) / stations.length;
    
    animateValue('metric-aqi', 0, avgAqi, 1000, 1);
    
    const aqiBadge = document.getElementById('aqi-badge');
    aqiBadge.textContent = getAqiLabel(avgAqi);
    aqiBadge.className = 'card-badge ' + getAqiBadgeClass(avgAqi);
    
    // Average pollutants
    const avgNo2 = stations.filter(s => s.no2 !== null).reduce((sum, s) => sum + s.no2, 0) / stations.filter(s => s.no2 !== null).length || 0;
    const avgO3 = stations.filter(s => s.o3 !== null).reduce((sum, s) => sum + s.o3, 0) / stations.filter(s => s.o3 !== null).length || 0;
    const avgPm10 = stations.filter(s => s.pm10 !== null).reduce((sum, s) => sum + s.pm10, 0) / stations.filter(s => s.pm10 !== null).length || 0;
    const avgPm25 = stations.filter(s => s.pm25 !== null).reduce((sum, s) => sum + s.pm25, 0) / stations.filter(s => s.pm25 !== null).length || 0;
    
    document.getElementById('metric-no2').textContent = avgNo2 > 0 ? avgNo2.toFixed(1) : '--';
    document.getElementById('metric-o3').textContent = avgO3 > 0 ? avgO3.toFixed(1) : '--';
    document.getElementById('metric-pm10').textContent = avgPm10 > 0 ? avgPm10.toFixed(1) : '--';
    document.getElementById('metric-pm25').textContent = avgPm25 > 0 ? avgPm25.toFixed(1) : '--';
}

// Update Weather
function updateWeather(weather) {
    if (!weather) return;
    
    const card = document.getElementById('weather-card');
    card.style.display = 'block';
    
    if (weather.temperature !== null) {
        animateValue('weather-temp', 0, weather.temperature, 800, 1);
        document.getElementById('weather-temp').textContent += '°C';
    }
    if (weather.humidity !== null) {
        document.getElementById('weather-humidity').textContent = weather.humidity.toFixed(0) + '%';
    }
    if (weather.wind_speed !== null) {
        document.getElementById('weather-wind').textContent = weather.wind_speed.toFixed(1) + ' km/h';
    }
    if (weather.description) {
        document.getElementById('weather-desc').textContent = weather.description;
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
        
        item.innerHTML = `
            <div class="station-info">
                <div class="station-name">${station.station_name}</div>
                <div class="station-meta">${station.station_type}</div>
            </div>
            <div class="station-stats">
                <div class="station-stat">
                    <div class="station-stat-value">${(station.prob_good * 100).toFixed(0)}%</div>
                    <div class="station-stat-label">Probabilidad</div>
                </div>
                <div class="station-stat">
                    <div class="station-stat-value">${station.aqi.toFixed(1)}</div>
                    <div class="station-stat-label">AQI</div>
                </div>
                <span class="station-badge ${station.is_good_to_run ? 'badge-success' : 'badge-danger'}">
                    ${station.is_good_to_run ? '✅' : '❌'}
                </span>
            </div>
        `;
        
        item.addEventListener('click', () => {
            map.setView([station.latitude, station.longitude], 14);
            showToast(`Centrado en ${station.station_name}`, 'success');
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
    document.getElementById('loading-overlay').classList.remove('hidden');
}

function hideLoading() {
    document.getElementById('loading-overlay').classList.add('hidden');
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
