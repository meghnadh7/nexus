{{/*
Common label helpers for Nexus Helm chart.
*/}}

{{- define "nexus.labels" -}}
app.kubernetes.io/part-of: nexus
app.kubernetes.io/managed-by: {{ .Release.Service }}
helm.sh/chart: {{ .Chart.Name }}-{{ .Chart.Version }}
{{- end }}

{{- define "nexus.selectorLabels" -}}
app.kubernetes.io/name: {{ .name }}
app.kubernetes.io/instance: {{ .Release.Name }}
{{- end }}

{{- define "nexus.imageTag" -}}
{{ .Values.global.imageTag | default "latest" }}
{{- end }}

{{- define "nexus.image" -}}
{{ .Values.global.registry }}/{{ .repository }}:{{ include "nexus.imageTag" . }}
{{- end }}

{{- define "nexus.secretRef" -}}
- secretRef:
    name: nexus-secrets
{{- end }}
