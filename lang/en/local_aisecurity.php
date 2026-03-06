<?php
defined('MOODLE_INTERNAL') || die();

$string['pluginname'] = 'AI Based Anomaly Detection';
$string['description'] = 'AI Based Anomaly Detection plugin for Moodle. Monitors user logins and detects anomalies using a Python backend.';
$string['attackdetected'] = 'Attack Detected!';
$string['pluginsettings'] = 'Plugin Settings';
$string['enable'] = 'Enable AI Security';
$string['enable_desc'] = 'Enable or disable the AI anomaly detection.';
$string['apiurl'] = 'Python API URL';
$string['apiurl_desc'] = 'The URL of the Python Flask API (e.g., http://127.0.0.1:5000/predict).';
$string['sensitivity'] = 'Detection Sensitivity';
$string['sensitivity_desc'] = 'Adjust how strict the AI should be. Low = only block very confident threats (fewer false positives). High = block any suspected threat (fewer false negatives).';
$string['sensitivity_low']    = 'Low — block if confidence ≥ 90%';
$string['sensitivity_medium'] = 'Medium — block if confidence ≥ 70% (recommended)';
$string['sensitivity_high']   = 'High — block if confidence ≥ 50%';
$string['headerconfig'] = 'AI Security Configuration';
