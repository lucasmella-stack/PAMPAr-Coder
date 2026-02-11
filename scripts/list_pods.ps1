# Full pod status check
$apiKey = "$env:RUNPOD_API_KEY"
$body='{"query":"query{myself{pods{id name desiredStatus runtime{uptimeInSeconds}}}}"}'
$r=Invoke-RestMethod -Uri "https://api.runpod.io/graphql?api_key=$apiKey" -Method POST -Headers @{"Content-Type"="application/json"} -Body $body -TimeoutSec 30
$r.data.myself.pods | ForEach-Object {
    $up = if($_.runtime){"up:$($_.runtime.uptimeInSeconds)s"}else{"no-runtime"}
    "$($_.id)|$($_.name)|$($_.desiredStatus)|$up"
} | Out-File C:\Users\lucas\Documents\pods_status.txt
