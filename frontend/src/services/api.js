const API_BASE = import.meta.env.VITE_API_BASE_URL || 'http://localhost:8000/api/v1';

export async function getIntegrationReadiness() {
  const response = await fetch(`${API_BASE}/integration-readiness`);
  if (!response.ok) throw new Error(`Readiness request failed: ${response.status}`);
  return response.json();
}
