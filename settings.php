<?php
defined('MOODLE_INTERNAL') || die();

if ($hassiteconfig) {
    $settings = new admin_settingpage('local_aisecurity', get_string('pluginname', 'local_aisecurity'));

    $settings->add(new admin_setting_heading(
        'local_aisecurity/config',
        get_string('headerconfig', 'local_aisecurity'),
        ''
    ));

    // Enable/Disable Checkbox
    $settings->add(new admin_setting_configcheckbox(
        'local_aisecurity/enable',
        get_string('enable', 'local_aisecurity'),
        get_string('enable_desc', 'local_aisecurity'),
        1
    ));

    // API URL Text Input
    $settings->add(new admin_setting_configtext(
        'local_aisecurity/apiurl',
        get_string('apiurl', 'local_aisecurity'),
        get_string('apiurl_desc', 'local_aisecurity'),
        'http://127.0.0.1:5001/predict',
        PARAM_URL
    ));

    // Sensitivity Select
    $settings->add(new admin_setting_configselect(
        'local_aisecurity/sensitivity',
        get_string('sensitivity', 'local_aisecurity'),
        get_string('sensitivity_desc', 'local_aisecurity'),
        '2',
        [
            '1' => get_string('sensitivity_low', 'local_aisecurity'),
            '2' => get_string('sensitivity_medium', 'local_aisecurity'),
            '3' => get_string('sensitivity_high', 'local_aisecurity'),
        ]
    ));

    $ADMIN->add('localplugins', $settings);
}
