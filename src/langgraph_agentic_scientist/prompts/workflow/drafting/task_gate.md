
# Harness-owned validation gate
After each edit, run the complete acceptance check. Call the `task_gate` tool:

    task_gate(check="acceptance-smoke")

The production-entry check uses ordinary production device discovery and requires one completed
declared reset in each isolated plane. The service is fingerprint-bound, allowlisted, and does
not expose Docker or sealed final values.
Passing an in-session check is advisory; the harness repeats all gates after you exit. Address
every issue in the current repair prompt, then stop once their relevant gates pass. Do not spend
the bounded request budget rerunning unrelated gates; a refused request after the service closes
is operational, not a new scaffold failure.
