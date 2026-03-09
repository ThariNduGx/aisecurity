<?php
namespace local_aisecurity;

defined('MOODLE_INTERNAL') || die();

class observer {
    /**
     * Handle user_loggedin event.
     */
    public static function user_loggedin(\core\event\user_loggedin $event) {
        global $CFG;

        // 1. Check if Plugin is Enabled
        $enabled = get_config('local_aisecurity', 'enable');
        if (!$enabled) {
            return;
        }

        // 2. Get Data
        $userid = $event->objectid;
        $ip = $event->other['ip'] ?? $_SERVER['REMOTE_ADDR'];

        // 3. Call Python API
        require_once($CFG->dirroot . '/local/aisecurity/lib.php');
        $response = local_aisecurity_check_ip($userid, $ip);

        // 4. Read sensitivity setting and calculate threshold
        $sensitivity = (int) get_config('local_aisecurity', 'sensitivity');
        $thresholds  = [1 => 0.90, 2 => 0.70, 3 => 0.50];
        $threshold   = $thresholds[$sensitivity] ?? 0.70;

        $is_attack  = isset($response['prediction']) && $response['prediction'] == 1;
        $confidence = isset($response['confidence']) ? (float) $response['confidence'] : 1.0;

        // 5. Analyze & Enforce
        if ($is_attack && $confidence >= $threshold) {

            // LOGGING — write to PHP error log (visible in XAMPP's php_error_log)
            $reason  = $response['reason'] ?? 'ml_model_detection';
            $logline = date('Y-m-d H:i:s') . " [AI-SECURITY] BLOCKED userid=$userid ip=$ip "
                     . "confidence=$confidence threshold=$threshold reason=$reason";
            error_log($logline);

            // CUSTOM DARK MODE "CYBER" SECURITY PAGE
            $html = '
            <!DOCTYPE html>
            <html lang="en">
            <head>
                <meta charset="UTF-8">
                <meta name="viewport" content="width=device-width, initial-scale=1.0">
                <title>Security Alert</title>
                <style>
                    :root {
                        --primary-red: #ff3b3b;
                        --glow-red: rgba(255, 59, 59, 0.4);
                        --bg-dark: #0a0a0a;
                        --card-bg: rgba(20, 20, 20, 0.8);
                    }
                    body {
                        background-color: var(--bg-dark);
                        background-image: radial-gradient(circle at 50% 50%, #2a0a0a 0%, #000000 100%);
                        font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Ubuntu, sans-serif;
                        display: flex;
                        justify-content: center;
                        align-items: center;
                        height: 100vh;
                        margin: 0;
                        color: #ffffff;
                        overflow: hidden;
                    }
                    .security-card {
                        background: var(--card-bg);
                        backdrop-filter: blur(20px);
                        -webkit-backdrop-filter: blur(20px);
                        padding: 3.5rem;
                        border-radius: 1.5rem;
                        border: 1px solid rgba(255, 255, 255, 0.1);
                        box-shadow: 0 0 50px var(--glow-red), inset 0 0 20px rgba(0, 0, 0, 0.5);
                        text-align: center;
                        max-width: 500px;
                        width: 90%;
                        position: relative;
                        animation: zoomIn 0.5s ease-out;
                    }
                    .security-card::before {
                        content: "";
                        position: absolute;
                        top: 0; left: 0; right: 0; height: 4px;
                        background: linear-gradient(90deg, transparent, var(--primary-red), transparent);
                        border-radius: 1.5rem 1.5rem 0 0;
                    }
                    .icon {
                        font-size: 5rem;
                        margin-bottom: 1.5rem;
                        display: block;
                        text-shadow: 0 0 30px var(--primary-red);
                        animation: pulse 2s infinite;
                    }
                    h1 {
                        color: white;
                        font-size: 2.2rem;
                        font-weight: 800;
                        margin-bottom: 1rem;
                        margin-top: 0;
                        letter-spacing: -0.5px;
                    }
                    p {
                        font-size: 1.15rem;
                        line-height: 1.6;
                        color: #d1d5db;
                        margin-bottom: 2rem;
                    }
                    .info-box {
                        background: rgba(255, 59, 59, 0.1);
                        border: 1px solid rgba(255, 59, 59, 0.2);
                        padding: 1rem;
                        border-radius: 0.75rem;
                        font-family: "SF Mono", "Monoco", "Consolas", monospace;
                        font-size: 0.9rem;
                        color: #ff8080;
                        margin-bottom: 2rem;
                        text-align: left;
                    }
                    .info-row {
                        display: flex;
                        justify-content: space-between;
                        margin-bottom: 0.5rem;
                    }
                    .info-row:last-child { margin-bottom: 0; }
                    .footer {
                        font-size: 0.8rem;
                        color: #6b7280;
                        text-transform: uppercase;
                        letter-spacing: 2px;
                        margin-top: 2rem;
                    }
                    
                    @keyframes pulse {
                        0% { transform: scale(1); opacity: 1; }
                        50% { transform: scale(1.05); opacity: 0.8; }
                        100% { transform: scale(1); opacity: 1; }
                    }
                    @keyframes zoomIn {
                        from { transform: scale(0.9); opacity: 0; }
                        to { transform: scale(1); opacity: 1; }
                    }
                </style>
            </head>
            <body>
                <div class="security-card">
                    <span class="icon">🚫</span>
                    <h1>Access Denied</h1>
                    <p>
                        Our AI Threat Detection System has intercepted a potential security anomaly from your connection. 
                        Your session has been terminated immediately.
                    </p>
                    <div class="info-box">
                        <div class="info-row">
                            <span>STATUS:</span>
                            <span style="font-weight: bold; color: #ff3b3b;">CRITICAL THREAT</span>
                        </div>
                        <div class="info-row">
                            <span>IP ADDRESS:</span>
                            <span>' . htmlspecialchars($ip) . '</span>
                        </div>
                        <div class="info-row">
                            <span>DETECTION ID:</span>
                            <span>' . strtoupper(uniqid()) . '</span>
                        </div>
                    </div>
                    <div class="footer">
                        Moodle AI Security Guard
                    </div>
                </div>
            </body>
            </html>';

            // TERMINATE SESSION SAFELY
            \core\session\manager::terminate_current();

            // DISPLAY BLOCK SCREEN AND STOP EXECUTION
            die($html);
        }
    }
}
