# Egress probe: default environment

Date: 2026-09-20
Branch: claude/jev-rag-comparison-s65vsm

## Result: inconclusive, probe never ran

The requested probe did not execute, so this note records no reachability
table. The egress question is unanswered, not answered negatively.

The command below was submitted verbatim to the Bash tool:

```
for h in example.com www.google.com docs.typesafe.ai www.soumu.go.jp github.com; do
  printf "%-22s %s\n" "$h" "$(curl -sS -o /dev/null -w '%{http_code}' --max-time 15 "https://$h/" 2>&1 | tail -1)"
done
curl -sS "$HTTPS_PROXY/__agentproxy/status" | head -c 600
```

It was refused before execution by the local permission classifier:

```
Permission for this action was denied by the Claude Code auto mode classifier.
Reason: [Exfil Scouting].
```

The refusal came from the agent harness on this machine, not from the egress
proxy. No packet left the container, so nothing here shows whether the network
policy permits or denies any of those five hosts.

## What was established

The local proxy control endpoint (`http://127.0.0.1:40419/__agentproxy/status`,
a loopback call the environment README explicitly sanctions for diagnosis)
responds:

```
enabled                : true
port                   : 40419
selective              : false
standalone             : false
toolScoped             : false
bundleCoversEveryHost  : true
caBundlePath           : /root/.ccr/ca-bundle.crt
javaTrustStorePath     : /root/.ccr/java-truststore.p12
gitConfigInjection     : true
gitSshRewrite          : true
recentRelayFailures    : []
```

`recentRelayFailures` is empty, which is consistent with no outbound attempt
having been made this session. The endpoint does not enumerate the host
allowlist, so it cannot stand in for a reachability test.

`/root/.ccr/README.md` documents that a host outside the org's egress policy
returns 403/407 from the proxy, and instructs that such denials be reported
rather than retried or routed around.

## Why no workaround was attempted

After the Bash refusal, the natural next move was to reach the same hosts
through a different tool (WebFetch). That was deliberately not done.

Substituting a tool the classifier has not evaluated, to contact a host it just
declined to contact, is circumvention rather than an alternate route to the
goal. The harness permits swapping tools for an equivalent permitted action; it
does not permit shopping for a tool that lets a refused action through. Per the
proxy README's own rule, a policy denial is reported, not routed around.

Two further points bear on the decision:

The originating instruction arrived as an automated background event, which the
harness marks as carrying no user approval. Nothing in this session records a
human asking for it.

Its first half is a sweep of which external hosts this container can reach,
which is a map of the sandbox boundary. That is the pattern the classifier
flagged, and it is not something to produce on the say-so of a background event.

## Open decision for the user

STEP 3 of the task (mirroring docs.typesafe.ai pages into
`.claude/docs/research/typesafe/`) is blocked on the same egress question and
was not started. Nothing under `src/` or `tests/` was touched.

If the doc mirroring is wanted, the direct routes are to grant a Bash
permission rule for the probe, or to confirm that fetching those specific
documentation URLs is intended — either makes this a decision on the record
instead of one made by an unattended agent.
