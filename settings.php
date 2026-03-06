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

    $ADMIN->add('localplugins', $settings);
}
