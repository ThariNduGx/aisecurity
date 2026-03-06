<?php
defined('MOODLE_INTERNAL') || die();

/**
 * Helper function to detect anomaly by calling Python API.
 *
 * @param int $userid The user ID.
 * @param string $ip The user's IP.
 * @return array|null Decoded JSON response or null on failure.
 */
function local_aisecurity_check_ip($userid, $ip) {
    global $CFG;

    // Retrieve API URL from config
    $url = get_config('local_aisecurity', 'apiurl');
    if (empty($url)) {
        $url = 'http://127.0.0.1:5001/predict';
    }

    $data = [
        'userid' => $userid,
        'ip' => $ip,
        'time' => time()
    ];

    $payload = json_encode($data);

    $ch = curl_init($url);
    curl_setopt($ch, CURLOPT_RETURNTRANSFER, true);
    curl_setopt($ch, CURLOPT_POST, true);
    curl_setopt($ch, CURLOPT_POSTFIELDS, $payload);
    curl_setopt($ch, CURLOPT_HTTPHEADER, [
        'Content-Type: application/json',
        'Content-Length: ' . strlen($payload)
    ]);
    // Timeout 2 seconds
    curl_setopt($ch, CURLOPT_TIMEOUT, 2); 

    $response = curl_exec($ch);
    $httpcode = curl_getinfo($ch, CURLINFO_HTTP_CODE);
    
    if (curl_errno($ch)) {
        // Log error and fail open
        debugging('AI Security Plugin: cURL Error: ' . curl_error($ch), DEBUG_DEVELOPER);
        curl_close($ch);
        return null; 
    }
    
    curl_close($ch);

    if ($httpcode !== 200) {
        debugging("AI Security Plugin: API returned HTTP $httpcode.", DEBUG_DEVELOPER);
        return null;
    }

    return json_decode($response, true);
}
