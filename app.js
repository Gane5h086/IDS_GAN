const scenarioDatasetMap = {
  dos: 'nsl_kdd',
  probe: 'nsl_kdd',
  r2l: 'nsl_kdd',
  u2r: 'unsw_nb15',
  brute_force: 'cicids2017'
};

function el(id) {
  return document.getElementById(id);
}

function baseUrl() {
  return el('defender-url').value.trim().replace(/\/$/, '');
}

function selectedDataset() {
  return el('dataset').value;
}

function formatTime(timestamp) {
  const date = new Date(timestamp * 1000);
  return date.toLocaleTimeString();
}

async function sendScenario(scenario) {
  const scenarioDataset = scenarioDatasetMap[scenario] || selectedDataset();
  el('dataset').value = scenarioDataset;

  const payload = {
    dataset: scenarioDataset,
    scenario,
    source_ip: el('source-ip').value.trim(),
    destination_ip: el('destination-ip').value.trim()
  };

  el('send-status').textContent = `Sending ${scenario} attack...`;

  const response = await fetch(`${baseUrl()}/demo_emit`, {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(payload)
  });

  const data = await response.json();
  if (data.status !== 'success') {
    throw new Error(data.message || 'Attack failed to send');
  }

  // Attacker doesn't see defender's predictions or event details
  el('send-status').textContent = `${scenario} attack sent at ${new Date().toLocaleTimeString()}.`;
}

window.addEventListener('DOMContentLoaded', () => {
  document.querySelectorAll('[data-scenario]').forEach((button) => {
    button.addEventListener('click', async () => {
      try {
        await sendScenario(button.dataset.scenario);
      } catch (error) {
        el('send-status').textContent = `Error: ${error.message}`;
      }
    });
  });
});
