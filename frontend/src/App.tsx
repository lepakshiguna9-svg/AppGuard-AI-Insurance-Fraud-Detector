import React, { useState, useEffect } from 'react';

interface FeatureDeviation {
  feature_name: string;
  feature_type: string;
  submitted_value: any;
  reconstructed_value: any;
  reconstruction_error: number;
  statistical_observation: string;
}

interface ScoredApplication {
  application_id: string;
  policy_type: string;
  coverage_amount: number;
  deductible_amount: number;
  premium_frequency: string;
  application_channel: string;
  applicant_age: number;
  annual_income_estimate: number;
  prior_application_count: number;
  recent_address_changes_count: number;
  property_or_vehicle_age: number;
  beneficiary_relation: string;
  declared_credit_tier: string;
  // Scoring results
  anomaly_score: number;
  calibrated_threshold: number;
  review_recommended: boolean;
  review_priority: string;
  neutral_summary: string;
  top_deviations: FeatureDeviation[];
  // Local review workflow status
  adjudication_state?: 'pending' | 'cleared' | 'docs_requested' | 'siu_referral';
  underwriter_notes?: string;
  scenario_tag?: string;
}

interface ModelInfo {
  artifact_version: string;
  model_architecture: string;
  calibrated_threshold: number;
  target_percentile: number;
  input_dimension: number;
  latent_dimension: number;
}

export function App() {
  const [applications, setApplications] = useState<ScoredApplication[]>([]);
  const [selectedAppId, setSelectedAppId] = useState<string | null>(null);
  const [filterState, setFilterState] = useState<'all' | 'review' | 'standard'>('all');
  const [searchQuery, setSearchQuery] = useState('');
  const [modelInfo, setModelInfo] = useState<ModelInfo | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [isScoringManual, setIsScoringManual] = useState(false);
  const [manualModalOpen, setManualModalOpen] = useState(false);
  const [manualFormError, setManualFormError] = useState<string | null>(null);
  const [jsonModalOpen, setJsonModalOpen] = useState(false);
  const [jsonInputText, setJsonInputText] = useState('');
  const [jsonError, setJsonError] = useState<string | null>(null);
  const [isUploadingJson, setIsUploadingJson] = useState(false);
  const [statusMessage, setStatusMessage] = useState<string | null>(null);

  // Manual form state
  const [manualForm, setManualForm] = useState({
    policy_type: 'auto',
    coverage_amount: 50000,
    deductible_amount: 500,
    premium_frequency: 'monthly',
    application_channel: 'online_direct',
    applicant_age: 34,
    annual_income_estimate: 72000,
    prior_application_count: 0,
    recent_address_changes_count: 0,
    property_or_vehicle_age: 4,
    beneficiary_relation: 'spouse',
    declared_credit_tier: 'good',
  });

  // Fetch model metadata and initial demo samples on mount
  useEffect(() => {
    fetchModelInfo();
    loadCuratedSamples();
  }, []);

  const fetchModelInfo = async () => {
    try {
      const res = await fetch('/api/v1/model-info');
      if (res.ok) {
        const data = await res.json();
        setModelInfo(data);
      }
    } catch (err) {
      console.warn('Backend not yet reachable, using defaults');
      setModelInfo({
        artifact_version: '1.0.0',
        model_architecture: 'TabularVAE',
        calibrated_threshold: 0.1542,
        target_percentile: 95.0,
        input_dimension: 31,
        latent_dimension: 8,
      });
    }
  };

  const loadCuratedSamples = async () => {
    setIsLoading(true);
    setStatusMessage('Loading curated synthetic scenarios...');
    try {
      const res = await fetch('/api/v1/demo/samples');
      if (res.ok) {
        const data = await res.json();
        const scoredList: ScoredApplication[] = [];

        for (const sample of data.samples) {
          const scoreRes = await fetch('/api/v1/score', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify(sample),
          });

          if (scoreRes.ok) {
            const scoreData = await scoreRes.json();
            scoredList.push({
              ...sample,
              anomaly_score: scoreData.anomaly_score,
              calibrated_threshold: scoreData.calibrated_threshold,
              review_recommended: scoreData.review_recommended,
              review_priority: scoreData.review_priority,
              neutral_summary: scoreData.neutral_summary,
              top_deviations: scoreData.top_deviations || [],
              adjudication_state: 'pending',
              scenario_tag: sample.description,
            });
          }
        }

        setApplications(scoredList);
        if (scoredList.length > 0) {
          setSelectedAppId(scoredList[0].application_id);
        }
        setStatusMessage(null);
      }
    } catch (err) {
      console.error(err);
      setStatusMessage('Error contacting scoring backend. Ensure server is active on port 8000.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleGenerateBatch = async () => {
    setIsLoading(true);
    setStatusMessage('Synthesizing batch of 15 applications with realistic distributions...');
    try {
      const res = await fetch('/api/v1/demo/generate-batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ count: 15, include_anomalies: true }),
      });

      if (res.ok) {
        const data = await res.json();
        const mapped: ScoredApplication[] = data.scored_applications.map((item: any, idx: number) => {
          const raw = data.raw_applications[idx];
          return {
            ...raw,
            application_id: item.application_id || `APP-${idx + 1}`,
            anomaly_score: item.anomaly_score,
            calibrated_threshold: item.calibrated_threshold || modelInfo?.calibrated_threshold || 0.1542,
            review_recommended: item.review_recommended,
            review_priority: item.review_priority || (item.review_recommended ? 'elevated' : 'standard'),
            neutral_summary: item.neutral_summary,
            top_deviations: item.top_deviations || [],
            adjudication_state: 'pending',
          };
        });

        setApplications(mapped);
        if (mapped.length > 0) {
          setSelectedAppId(mapped[0].application_id);
        }
        setStatusMessage(null);
      }
    } catch (err) {
      setStatusMessage('Failed to generate synthetic batch.');
    } finally {
      setIsLoading(false);
    }
  };

  const handleManualSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setIsScoringManual(true);
    setManualFormError(null);
    try {
      const payload = {
        ...manualForm,
        application_id: `CUSTOM-${Date.now().toString().slice(-4)}`,
        coverage_amount: Number(manualForm.coverage_amount),
        deductible_amount: Number(manualForm.deductible_amount),
        applicant_age: Number(manualForm.applicant_age),
        annual_income_estimate: Number(manualForm.annual_income_estimate),
        prior_application_count: Number(manualForm.prior_application_count),
        recent_address_changes_count: Number(manualForm.recent_address_changes_count),
        property_or_vehicle_age: Number(manualForm.property_or_vehicle_age),
      };

      const res = await fetch('/api/v1/score', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(payload),
      });

      if (res.ok) {
        const scoreData = await res.json();
        const newApp: ScoredApplication = {
          ...payload,
          anomaly_score: scoreData.anomaly_score,
          calibrated_threshold: scoreData.calibrated_threshold,
          review_recommended: scoreData.review_recommended,
          review_priority: scoreData.review_priority,
          neutral_summary: scoreData.neutral_summary,
          top_deviations: scoreData.top_deviations || [],
          adjudication_state: 'pending',
          scenario_tag: 'Interactive Underwriter Submission',
        };

        setApplications([newApp, ...applications]);
        setSelectedAppId(newApp.application_id);
        setManualFormError(null);
        setManualModalOpen(false);
      } else {
        const errData = await res.json().catch(() => ({}));
        const detailMsg = typeof errData.detail === 'string'
          ? errData.detail
          : (Array.isArray(errData.detail)
              ? errData.detail.map((d: any) => `${d.loc?.slice(1).join('.') || 'field'}: ${d.msg}`).join('; ')
              : 'Validation failed.');
        setManualFormError(`Validation error: ${detailMsg}`);
      }
    } catch (err: any) {
      setManualFormError(err.message || 'Failed to connect to scoring API.');
    } finally {
      setIsScoringManual(false);
    }
  };

  const handleJsonUpload = async (e: React.FormEvent) => {
    e.preventDefault();
    setJsonError(null);
    setIsUploadingJson(true);

    try {
      if (!jsonInputText.trim()) {
        throw new Error('Please enter JSON application data or select a .json file.');
      }

      let parsed: any;
      try {
        parsed = JSON.parse(jsonInputText);
      } catch (err: any) {
        throw new Error(`JSON Syntax Error: ${err.message}`);
      }

      const recordsArray: any[] = Array.isArray(parsed) ? parsed : [parsed];

      if (recordsArray.length === 0) {
        throw new Error('JSON payload contains no application records.');
      }

      if (recordsArray.length > 100) {
        throw new Error(`Batch contains ${recordsArray.length} records. Maximum allowed batch size is 100 applications.`);
      }

      const res = await fetch('/api/v1/score/batch', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ applications: recordsArray }),
      });

      if (!res.ok) {
        const errData = await res.json().catch(() => ({}));
        const detailMsg = typeof errData.detail === 'string'
          ? errData.detail
          : (Array.isArray(errData.detail)
              ? errData.detail.map((d: any) => `${d.loc?.join('.')}: ${d.msg}`).join('; ')
              : 'Batch scoring request rejected.');
        throw new Error(`API Error (${res.status}): ${detailMsg}`);
      }

      const batchData = await res.json();
      const newScoredList: ScoredApplication[] = [];
      const failedIndices: { index: number; msg: string }[] = [];

      batchData.results.forEach((r: any, idx: number) => {
        const raw = recordsArray[idx] || {};
        if (r.status === 'success') {
          newScoredList.push({
            ...raw,
            application_id: r.application_id || raw.application_id || `UPLOAD-${idx + 1}`,
            anomaly_score: r.anomaly_score,
            calibrated_threshold: r.calibrated_threshold || modelInfo?.calibrated_threshold || 0.1542,
            review_recommended: r.review_recommended,
            review_priority: r.review_priority || (r.review_recommended ? 'elevated' : 'standard'),
            neutral_summary: r.neutral_summary,
            top_deviations: r.top_deviations || [],
            adjudication_state: 'pending',
            scenario_tag: 'Uploaded Batch Record',
          });
        } else {
          failedIndices.push({ index: idx, msg: r.error_message || 'Validation error' });
        }
      });

      if (newScoredList.length > 0) {
        setApplications(prev => [...newScoredList, ...prev]);
        setSelectedAppId(newScoredList[0].application_id);
      }

      if (failedIndices.length > 0) {
        const summary = `Scored ${newScoredList.length} of ${recordsArray.length} records. ` +
          `${failedIndices.length} record(s) failed validation:\n` +
          failedIndices.map(f => `• Record #${f.index + 1}: ${f.msg}`).slice(0, 3).join('\n') +
          (failedIndices.length > 3 ? `\n...and ${failedIndices.length - 3} more` : '');
        setJsonError(summary);
      } else {
        setJsonModalOpen(false);
        setJsonInputText('');
        setStatusMessage(`Successfully scored and loaded ${newScoredList.length} application(s).`);
      }
    } catch (err: any) {
      setJsonError(err.message || 'Failed to process JSON batch.');
    } finally {
      setIsUploadingJson(false);
    }
  };

  const loadExampleJsonTemplate = () => {
    const template = [
      {
        "application_id": "UPLOAD-DEMO-01",
        "policy_type": "auto",
        "coverage_amount": 60000,
        "deductible_amount": 500,
        "premium_frequency": "monthly",
        "application_channel": "online_direct",
        "applicant_age": 35,
        "annual_income_estimate": 75000,
        "prior_application_count": 0,
        "recent_address_changes_count": 0,
        "property_or_vehicle_age": 3,
        "beneficiary_relation": "spouse",
        "declared_credit_tier": "good"
      },
      {
        "application_id": "UPLOAD-DEMO-02",
        "policy_type": "term_life",
        "coverage_amount": 2500000,
        "deductible_amount": 0,
        "premium_frequency": "monthly",
        "application_channel": "online_direct",
        "applicant_age": 22,
        "annual_income_estimate": 19000,
        "prior_application_count": 0,
        "recent_address_changes_count": 1,
        "property_or_vehicle_age": 0,
        "beneficiary_relation": "other",
        "declared_credit_tier": "limited_history"
      }
    ];
    setJsonInputText(JSON.stringify(template, null, 2));
    setJsonError(null);
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (!file) return;
    const reader = new FileReader();
    reader.onload = (event) => {
      const content = event.target?.result as string;
      setJsonInputText(content);
      setJsonError(null);
    };
    reader.onerror = () => {
      setJsonError('Failed to read the selected file.');
    };
    reader.readAsText(file);
  };

  const updateAdjudication = (appId: string, state: ScoredApplication['adjudication_state'], notes?: string) => {
    setApplications(prev => prev.map(app => {
      if (app.application_id === appId) {
        return {
          ...app,
          adjudication_state: state,
          underwriter_notes: notes !== undefined ? notes : app.underwriter_notes,
        };
      }
      return app;
    }));
  };

  const selectedApp = applications.find(a => a.application_id === selectedAppId);

  // Filter applications
  const filteredApps = applications.filter(app => {
    if (filterState === 'review' && !app.review_recommended) return false;
    if (filterState === 'standard' && app.review_recommended) return false;
    if (searchQuery.trim()) {
      const q = searchQuery.toLowerCase();
      const matchId = app.application_id.toLowerCase().includes(q);
      const matchType = app.policy_type.toLowerCase().includes(q);
      const matchChannel = app.application_channel.toLowerCase().includes(q);
      return matchId || matchType || matchChannel;
    }
    return true;
  });

  const totalCount = applications.length;
  const reviewCount = applications.filter(a => a.review_recommended).length;
  const standardCount = totalCount - reviewCount;
  const activeThreshold = modelInfo?.calibrated_threshold ?? 0.1542;

  return (
    <div style={{ minHeight: '100vh', width: '100%', display: 'flex', flexDirection: 'column', background: 'var(--bg-page)' }}>
      {/* 1. Persistent Plain-Language Regulatory & Governance Notice Banner */}
      <aside
        role="note"
        aria-label="Regulatory and Operational Boundary Notice"
        style={{
          background: '#f6f3ea',
          borderBottom: '1px solid var(--border-medium)',
          padding: '7px 20px',
          fontSize: '11px',
          color: 'var(--text-body)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'space-between',
        }}
      >
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{
            background: 'var(--amber-dark)',
            color: '#fff',
            fontSize: '9px',
            fontWeight: 700,
            padding: '1px 5px',
            borderRadius: '2px',
            letterSpacing: '0.05em',
            textTransform: 'uppercase',
          }}>
            Operational Boundary
          </span>
          <span>
            <strong>Educational Demonstration Notice:</strong> Scores indicate statistical divergence from nominal application profiles only.
            Scores <em>do not prove fraud or deceptive intent</em>. Applicable regulatory and adverse-action requirements vary by jurisdiction and line of business;
            real deployments require qualified legal and compliance review. Scores must not be used for automated approval, denial, pricing, or adverse action.
            Human underwriting review is required. Not legal advice.
          </span>
        </div>
        <div style={{ display: 'flex', alignItems: 'center', gap: '12px', fontSize: '11px', color: 'var(--text-muted)' }}>
          <span>Demo Mode: 100% Synthetic Data</span>
          <span>Zero External Telemetry</span>
        </div>
      </aside>

      {/* 2. Quiet Editorial Workbench Header */}
      <header style={{
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-subtle)',
        padding: '14px 24px',
        display: 'flex',
        alignItems: 'center',
        justifyContent: 'space-between',
      }}>
        <div style={{ display: 'flex', alignItems: 'baseline', gap: '14px' }}>
          <h1 className="font-serif" style={{ fontSize: '20px', fontWeight: 600, letterSpacing: '-0.01em', color: 'var(--text-ink)' }}>
            AppGuard AI
          </h1>
          <span style={{ color: 'var(--text-muted)', fontSize: '13px' }}>
            Insurance Application Anomaly Review Workbench
          </span>
          <span style={{
            fontSize: '11px',
            color: 'var(--forest-dark)',
            background: 'var(--forest-light)',
            padding: '2px 8px',
            borderRadius: '2px',
            border: '1px solid var(--forest-border)',
            fontFamily: 'var(--font-mono)',
          }}>
            VAE Architecture v{modelInfo?.artifact_version || '1.0.0'}
          </span>
        </div>

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <button className="btn btn-sm" onClick={loadCuratedSamples} disabled={isLoading} title="Reload default curated synthetic scenarios">
            Curated Scenarios
          </button>
          <button className="btn btn-sm" onClick={handleGenerateBatch} disabled={isLoading} title="Synthesize new batch of 15 policies">
            Synthesize Batch (15)
          </button>
          <button className="btn btn-sm" onClick={() => { setJsonModalOpen(true); setJsonError(null); }} disabled={isLoading} title="Upload JSON application or batch">
            Upload JSON
          </button>
          <button className="btn btn-primary btn-sm" onClick={() => { setManualModalOpen(true); setManualFormError(null); }}>
            + Score Custom Application
          </button>
        </div>
      </header>

      {/* Status banner if message present */}
      {statusMessage && (
        <div style={{
          background: 'var(--slate-light)',
          padding: '6px 24px',
          fontSize: '12px',
          color: 'var(--slate-dark)',
          borderBottom: '1px solid var(--slate-border)',
        }}>
          {statusMessage}
        </div>
      )}

      {/* 3. Modest Operational Overview Bar */}
      <section style={{
        background: 'var(--bg-surface)',
        borderBottom: '1px solid var(--border-medium)',
        padding: '10px 24px',
        display: 'flex',
        alignItems: 'center',
        gap: '32px',
      }}>
        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)' }}>
            Queue Size
          </span>
          <span className="font-mono" style={{ fontSize: '15px', fontWeight: 600, color: 'var(--text-ink)' }}>
            {totalCount}
          </span>
        </div>

        <div style={{ height: '16px', width: '1px', background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)' }}>
            Standard Routing
          </span>
          <span className="font-mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--forest-dark)' }}>
            {standardCount}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            ({totalCount ? Math.round((standardCount / totalCount) * 100) : 0}%)
          </span>
        </div>

        <div style={{ height: '16px', width: '1px', background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)' }}>
            Review Recommended
          </span>
          <span className="font-mono" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--amber-dark)' }}>
            {reviewCount}
          </span>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
            ({totalCount ? Math.round((reviewCount / totalCount) * 100) : 0}%)
          </span>
        </div>

        <div style={{ height: '16px', width: '1px', background: 'var(--border-subtle)' }} />

        <div style={{ display: 'flex', alignItems: 'center', gap: '8px' }}>
          <span style={{ fontSize: '11px', textTransform: 'uppercase', letterSpacing: '0.04em', color: 'var(--text-muted)' }}>
            Calibrated 95th %ile Threshold
          </span>
          <span className="font-mono" style={{ fontSize: '13px', color: 'var(--text-body)', fontWeight: 500 }}>
            {activeThreshold.toFixed(4)}
          </span>
        </div>

        <div style={{ marginLeft: 'auto', display: 'flex', alignItems: 'center', gap: '10px' }}>
          <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>Filter View:</span>
          <div style={{ display: 'inline-flex', border: '1px solid var(--border-medium)', borderRadius: '3px', overflow: 'hidden' }}>
            <button
              onClick={() => setFilterState('all')}
              style={{
                padding: '3px 9px',
                fontSize: '11px',
                background: filterState === 'all' ? 'var(--forest-dark)' : '#fff',
                color: filterState === 'all' ? '#fff' : 'var(--text-body)',
                border: 'none',
                cursor: 'pointer',
              }}
            >
              All ({totalCount})
            </button>
            <button
              onClick={() => setFilterState('review')}
              style={{
                padding: '3px 9px',
                fontSize: '11px',
                background: filterState === 'review' ? 'var(--amber-dark)' : '#fff',
                color: filterState === 'review' ? '#fff' : 'var(--text-body)',
                border: 'none',
                borderLeft: '1px solid var(--border-medium)',
                cursor: 'pointer',
              }}
            >
              Review Flagged ({reviewCount})
            </button>
            <button
              onClick={() => setFilterState('standard')}
              style={{
                padding: '3px 9px',
                fontSize: '11px',
                background: filterState === 'standard' ? 'var(--forest-dark)' : '#fff',
                color: filterState === 'standard' ? '#fff' : 'var(--text-body)',
                border: 'none',
                borderLeft: '1px solid var(--border-medium)',
                cursor: 'pointer',
              }}
            >
              Standard ({standardCount})
            </button>
          </div>
        </div>
      </section>

      {/* 4. Workbench Main Content: Dual-Pane Layout */}
      <main className="workbench-main">
        {/* Left Pane: Queue Table */}
        <section className="workbench-left">
          {/* Table search & filter subheader */}
          <div style={{
            padding: '8px 16px',
            borderBottom: '1px solid var(--border-subtle)',
            background: 'var(--bg-surface-muted)',
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'space-between',
          }}>
            <input
              type="text"
              placeholder="Search by ID, policy type, channel..."
              value={searchQuery}
              onChange={e => setSearchQuery(e.target.value)}
              style={{
                padding: '4px 8px',
                fontSize: '11px',
                border: '1px solid var(--border-medium)',
                borderRadius: '2px',
                width: '240px',
                background: '#fff',
              }}
            />
            <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
              Showing {filteredApps.length} of {applications.length}
            </span>
          </div>

          {/* Table Content */}
          <div className="table-scroll-container">
            <table className="workbench-table" style={{ minWidth: '680px' }}>
              <thead>
                <tr>
                  <th style={{ width: '110px' }}>App ID</th>
                  <th>Policy Type</th>
                  <th style={{ textAlign: 'right' }}>Coverage</th>
                  <th style={{ textAlign: 'right' }}>Income</th>
                  <th>Channel</th>
                  <th>Age</th>
                  <th style={{ textAlign: 'right' }}>Score</th>
                  <th>Review Signal</th>
                  <th>Triage Status</th>
                </tr>
              </thead>
              <tbody>
                {filteredApps.length === 0 ? (
                  <tr>
                    <td colSpan={9} style={{ textAlign: 'center', padding: '32px', color: 'var(--text-muted)' }}>
                      No applications match the selected criteria.
                    </td>
                  </tr>
                ) : (
                  filteredApps.map(app => {
                    const isSelected = app.application_id === selectedAppId;
                    return (
                      <tr
                        key={app.application_id}
                        onClick={() => setSelectedAppId(app.application_id)}
                        className={isSelected ? 'selected' : ''}
                      >
                        <td className="font-mono" style={{ fontWeight: 600, color: 'var(--text-ink)' }}>
                          {app.application_id}
                        </td>
                        <td>
                          <span style={{ textTransform: 'capitalize' }}>
                            {app.policy_type.replace('_', ' ')}
                          </span>
                        </td>
                        <td className="font-mono" style={{ textAlign: 'right' }}>
                          ${app.coverage_amount.toLocaleString()}
                        </td>
                        <td className="font-mono" style={{ textAlign: 'right' }}>
                          ${app.annual_income_estimate.toLocaleString()}
                        </td>
                        <td style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                          {app.application_channel.replace('_', ' ')}
                        </td>
                        <td className="font-mono">{app.applicant_age}</td>
                        <td
                          className="font-mono"
                          style={{
                            textAlign: 'right',
                            fontWeight: 600,
                            color: app.review_recommended ? 'var(--amber-dark)' : 'var(--forest-dark)',
                          }}
                        >
                          {app.anomaly_score.toFixed(4)}
                        </td>
                        <td>
                          {app.review_recommended ? (
                            <span className={app.review_priority === 'high' ? 'badge badge-high' : 'badge badge-review'}>
                              Review Rec ({app.review_priority})
                            </span>
                          ) : (
                            <span className="badge badge-standard">
                              Standard Routing
                            </span>
                          )}
                        </td>
                        <td>
                          <span style={{
                            fontSize: '10px',
                            textTransform: 'uppercase',
                            letterSpacing: '0.04em',
                            color: app.adjudication_state === 'cleared' ? 'var(--forest-dark)' :
                                   app.adjudication_state === 'siu_referral' ? '#991b1b' :
                                   app.adjudication_state === 'docs_requested' ? 'var(--amber-dark)' : 'var(--text-dim)',
                            fontWeight: 600,
                          }}>
                            {app.adjudication_state ? app.adjudication_state.replace('_', ' ') : 'Pending'}
                          </span>
                        </td>
                      </tr>
                    );
                  })
                )}
              </tbody>
            </table>
          </div>
        </section>

        {/* Right Pane: Focused Application Detail & Statistical Deviation Panel */}
        <section className="workbench-right">
          {selectedApp ? (
            <div style={{ padding: '20px 24px', display: 'flex', flexDirection: 'column', gap: '20px' }}>
              {/* Detail Header */}
              <div style={{
                paddingBottom: '14px',
                borderBottom: '1px solid var(--border-medium)',
                display: 'flex',
                justifyContent: 'space-between',
                alignItems: 'flex-start',
              }}>
                <div>
                  <div style={{ display: 'flex', alignItems: 'center', gap: '10px', marginBottom: '4px' }}>
                    <h2 className="font-mono" style={{ fontSize: '18px', fontWeight: 600, color: 'var(--text-ink)' }}>
                      {selectedApp.application_id}
                    </h2>
                    {selectedApp.review_recommended ? (
                      <span className="badge badge-review">
                        Review Recommended ({selectedApp.review_priority})
                      </span>
                    ) : (
                      <span className="badge badge-standard">
                        Standard Underwriting
                      </span>
                    )}
                  </div>
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)' }}>
                    Policy Type: <strong style={{ color: 'var(--text-body)', textTransform: 'capitalize' }}>
                      {selectedApp.policy_type.replace('_', ' ')}
                    </strong> | Channel: {selectedApp.application_channel.replace('_', ' ')}
                  </div>
                </div>

                <div style={{ textAlign: 'right' }}>
                  <div style={{ fontSize: '11px', textTransform: 'uppercase', color: 'var(--text-muted)', letterSpacing: '0.04em' }}>
                    Reconstruction Error
                  </div>
                  <div
                    className="font-mono"
                    style={{
                      fontSize: '20px',
                      fontWeight: 600,
                      color: selectedApp.review_recommended ? 'var(--amber-dark)' : 'var(--forest-dark)',
                    }}
                  >
                    {selectedApp.anomaly_score.toFixed(4)}
                  </div>
                  <div style={{ fontSize: '10px', color: 'var(--text-dim)', fontFamily: 'var(--font-mono)' }}>
                    threshold: {selectedApp.calibrated_threshold.toFixed(4)}
                  </div>
                </div>
              </div>

              {/* Scenario tag if curated demo */}
              {selectedApp.scenario_tag && (
                <div style={{
                  padding: '8px 12px',
                  background: '#f4efe6',
                  borderRadius: '2px',
                  border: '1px solid var(--border-medium)',
                  fontSize: '11px',
                  color: 'var(--text-body)',
                }}>
                  <strong>Synthetic Scenario Description:</strong> {selectedApp.scenario_tag}
                </div>
              )}

              {/* Statistical Anomaly Gauge */}
              <div style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-subtle)',
                padding: '12px 16px',
                borderRadius: '3px',
              }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '11px', marginBottom: '6px' }}>
                  <span style={{ color: 'var(--text-muted)' }}>Statistical Deviation Relative to Baseline</span>
                  <span className="font-mono" style={{ color: 'var(--text-body)' }}>
                    {(selectedApp.anomaly_score / selectedApp.calibrated_threshold).toFixed(2)}x Threshold
                  </span>
                </div>
                <div style={{
                  height: '8px',
                  background: 'var(--bg-surface-muted)',
                  borderRadius: '4px',
                  position: 'relative',
                  overflow: 'hidden',
                  border: '1px solid var(--border-medium)',
                }}>
                  <div
                    style={{
                      position: 'absolute',
                      left: 0,
                      top: 0,
                      bottom: 0,
                      width: `${Math.min(100, (selectedApp.anomaly_score / (selectedApp.calibrated_threshold * 2)) * 100)}%`,
                      background: selectedApp.review_recommended ? 'var(--amber-base)' : 'var(--forest-base)',
                    }}
                  />
                  {/* Threshold marker */}
                  <div
                    style={{
                      position: 'absolute',
                      left: '50%',
                      top: 0,
                      bottom: 0,
                      width: '2px',
                      background: 'var(--text-ink)',
                    }}
                    title="Calibrated 95% Cutoff"
                  />
                </div>
                <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: '10px', color: 'var(--text-dim)', marginTop: '4px' }}>
                  <span>0.0 (Nominal)</span>
                  <span>Cutoff (95th %ile)</span>
                  <span>2.0x Cutoff</span>
                </div>
              </div>

              {/* Top Feature-Level Statistical Deviations */}
              <div>
                <h3 className="font-serif" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-ink)', marginBottom: '8px' }}>
                  Primary Feature-Level Statistical Deviations
                </h3>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                  Observed mathematical differences between submitted values and typical multidimensional manifold reconstruction.
                  These highlight areas for underwriting verification, not accusations of wrongdoing.
                </p>

                {selectedApp.top_deviations.length === 0 ? (
                  <div style={{ fontSize: '12px', color: 'var(--text-muted)', fontStyle: 'italic', padding: '12px', background: 'var(--bg-surface)' }}>
                    No significant statistical variances observed across feature dimensions.
                  </div>
                ) : (
                  <div style={{ display: 'flex', flexDirection: 'column', gap: '8px' }}>
                    {selectedApp.top_deviations.map((dev, idx) => (
                      <div
                        key={idx}
                        style={{
                          background: 'var(--bg-surface)',
                          border: '1px solid var(--border-subtle)',
                          borderRadius: '3px',
                          padding: '10px 14px',
                        }}
                      >
                        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', marginBottom: '4px' }}>
                          <span style={{ fontWeight: 600, fontSize: '12px', textTransform: 'capitalize' }}>
                            {dev.feature_name.replace(/_/g, ' ')}
                          </span>
                          <span className="font-mono" style={{ fontSize: '11px', color: 'var(--amber-dark)' }}>
                            Error: {dev.reconstruction_error.toFixed(4)}
                          </span>
                        </div>
                        <div style={{ fontSize: '11px', color: 'var(--text-body)', lineHeight: '1.4' }}>
                          {dev.statistical_observation}
                        </div>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* Submitted Profile Data */}
              <div>
                <h3 className="font-serif" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-ink)', marginBottom: '8px' }}>
                  Submitted Policy Application Data
                </h3>
                <div style={{
                  background: 'var(--bg-surface)',
                  border: '1px solid var(--border-subtle)',
                  borderRadius: '3px',
                  padding: '12px',
                  display: 'grid',
                  gridTemplateColumns: '1fr 1fr',
                  gap: '8px 16px',
                  fontSize: '11px',
                }}>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Coverage Amount: </span>
                    <strong className="font-mono">${selectedApp.coverage_amount.toLocaleString()}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Deductible: </span>
                    <strong className="font-mono">${selectedApp.deductible_amount.toLocaleString()}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Estimated Income: </span>
                    <strong className="font-mono">${selectedApp.annual_income_estimate.toLocaleString()}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Payment Schedule: </span>
                    <strong style={{ textTransform: 'capitalize' }}>{selectedApp.premium_frequency}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Applicant Age: </span>
                    <strong className="font-mono">{selectedApp.applicant_age} yrs</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Prior Apps (3 yrs): </span>
                    <strong className="font-mono">{selectedApp.prior_application_count}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Address Changes (2 yrs): </span>
                    <strong className="font-mono">{selectedApp.recent_address_changes_count}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Insured Asset Age: </span>
                    <strong className="font-mono">{selectedApp.property_or_vehicle_age} yrs</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Beneficiary: </span>
                    <strong style={{ textTransform: 'capitalize' }}>{selectedApp.beneficiary_relation}</strong>
                  </div>
                  <div>
                    <span style={{ color: 'var(--text-muted)' }}>Declared Credit: </span>
                    <strong style={{ textTransform: 'capitalize' }}>{selectedApp.declared_credit_tier}</strong>
                  </div>
                </div>
              </div>

              {/* Human Operational Adjudication Section */}
              <div style={{
                background: 'var(--bg-surface)',
                border: '1px solid var(--border-medium)',
                borderRadius: '3px',
                padding: '14px',
              }}>
                <h3 className="font-serif" style={{ fontSize: '14px', fontWeight: 600, color: 'var(--text-ink)', marginBottom: '8px' }}>
                  Operational Underwriter Adjudication
                </h3>
                <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '10px' }}>
                  Required governance step: An underwriter must document review rationale prior to binding.
                </p>

                <div style={{ display: 'flex', gap: '8px', flexWrap: 'wrap', marginBottom: '10px' }}>
                  <button
                    className="btn btn-sm"
                    style={{
                      background: selectedApp.adjudication_state === 'cleared' ? 'var(--forest-light)' : '#fff',
                      borderColor: selectedApp.adjudication_state === 'cleared' ? 'var(--forest-border)' : 'var(--border-medium)',
                      color: selectedApp.adjudication_state === 'cleared' ? 'var(--forest-dark)' : 'var(--text-body)',
                    }}
                    onClick={() => updateAdjudication(selectedApp.application_id, 'cleared')}
                  >
                    ✓ Clear for Standard Underwriting
                  </button>
                  <button
                    className="btn btn-sm"
                    style={{
                      background: selectedApp.adjudication_state === 'docs_requested' ? 'var(--amber-light)' : '#fff',
                      borderColor: selectedApp.adjudication_state === 'docs_requested' ? 'var(--amber-border)' : 'var(--border-medium)',
                      color: selectedApp.adjudication_state === 'docs_requested' ? 'var(--amber-dark)' : 'var(--text-body)',
                    }}
                    onClick={() => updateAdjudication(selectedApp.application_id, 'docs_requested')}
                  >
                    Request Additional Proof / Documents
                  </button>
                  <button
                    className="btn btn-sm"
                    style={{
                      background: selectedApp.adjudication_state === 'siu_referral' ? '#fee2e2' : '#fff',
                      borderColor: selectedApp.adjudication_state === 'siu_referral' ? '#fca5a5' : 'var(--border-medium)',
                      color: selectedApp.adjudication_state === 'siu_referral' ? '#991b1b' : 'var(--text-body)',
                    }}
                    onClick={() => updateAdjudication(selectedApp.application_id, 'siu_referral')}
                  >
                    Refer to SIU Investigation
                  </button>
                </div>

                <textarea
                  placeholder="Record underwriter review notes, verification findings, or justification..."
                  value={selectedApp.underwriter_notes || ''}
                  onChange={e => updateAdjudication(selectedApp.application_id, selectedApp.adjudication_state, e.target.value)}
                  style={{
                    width: '100%',
                    height: '60px',
                    padding: '6px 8px',
                    fontSize: '11px',
                    fontFamily: 'var(--font-sans)',
                    border: '1px solid var(--border-medium)',
                    borderRadius: '2px',
                    resize: 'vertical',
                  }}
                />
              </div>
            </div>
          ) : (
            <div style={{ textAlign: 'center', padding: '48px 24px', color: 'var(--text-muted)' }}>
              Select an application from the queue to inspect statistical deviations.
            </div>
          )}
        </section>
      </main>

      {/* 5. Interactive Manual Entry Modal / Drawer */}
      {manualModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(23, 25, 30, 0.45)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: 'var(--bg-surface)',
            width: '560px',
            maxHeight: '90vh',
            overflowY: 'auto',
            border: '1px solid var(--border-dark)',
            borderRadius: '4px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
            padding: '24px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 className="font-serif" style={{ fontSize: '18px', fontWeight: 600 }}>
                Score Custom Insurance Application
              </h2>
              <button
                className="btn btn-sm"
                onClick={() => setManualModalOpen(false)}
                style={{ border: 'none', background: 'transparent', fontSize: '16px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '16px' }}>
              Test how the calibrated VAE scores various field combinations. Try standard profiles or unusual combinations
              (e.g., $3,000,000 term life with $18,000 income, or high application velocity).
            </p>

            {manualFormError && (
              <div style={{
                padding: '8px 12px',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: '11px',
                borderRadius: '2px',
                marginBottom: '12px',
              }}>
                {manualFormError}
              </div>
            )}

            <form onSubmit={handleManualSubmit} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: '12px' }}>
                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Policy Type
                  </label>
                  <select
                    value={manualForm.policy_type}
                    onChange={e => setManualForm({ ...manualForm, policy_type: e.target.value })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  >
                    <option value="auto">Auto</option>
                    <option value="homeowners">Homeowners</option>
                    <option value="term_life">Term Life</option>
                    <option value="renters">Renters</option>
                    <option value="commercial_property">Commercial Property</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Requested Coverage ($ USD)
                  </label>
                  <input
                    type="number"
                    value={manualForm.coverage_amount}
                    onChange={e => setManualForm({ ...manualForm, coverage_amount: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Deductible ($ USD)
                  </label>
                  <input
                    type="number"
                    value={manualForm.deductible_amount}
                    onChange={e => setManualForm({ ...manualForm, deductible_amount: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Estimated Annual Income ($ USD)
                  </label>
                  <input
                    type="number"
                    value={manualForm.annual_income_estimate}
                    onChange={e => setManualForm({ ...manualForm, annual_income_estimate: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Applicant Age (yrs)
                  </label>
                  <input
                    type="number"
                    value={manualForm.applicant_age}
                    onChange={e => setManualForm({ ...manualForm, applicant_age: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Application Channel
                  </label>
                  <select
                    value={manualForm.application_channel}
                    onChange={e => setManualForm({ ...manualForm, application_channel: e.target.value })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  >
                    <option value="online_direct">Online Direct</option>
                    <option value="independent_broker">Independent Broker</option>
                    <option value="captive_agent">Captive Agent</option>
                    <option value="aggregator_partner">Aggregator Partner</option>
                    <option value="mobile_app">Mobile App</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Prior Apps Count (3 yrs)
                  </label>
                  <input
                    type="number"
                    value={manualForm.prior_application_count}
                    onChange={e => setManualForm({ ...manualForm, prior_application_count: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Address Changes (2 yrs)
                  </label>
                  <input
                    type="number"
                    value={manualForm.recent_address_changes_count}
                    onChange={e => setManualForm({ ...manualForm, recent_address_changes_count: Number(e.target.value) })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  />
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Premium Frequency
                  </label>
                  <select
                    value={manualForm.premium_frequency}
                    onChange={e => setManualForm({ ...manualForm, premium_frequency: e.target.value })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  >
                    <option value="monthly">Monthly</option>
                    <option value="quarterly">Quarterly</option>
                    <option value="semi_annual">Semi-Annual</option>
                    <option value="annual">Annual</option>
                  </select>
                </div>

                <div>
                  <label style={{ fontSize: '11px', fontWeight: 600, display: 'block', marginBottom: '4px' }}>
                    Declared Credit Tier
                  </label>
                  <select
                    value={manualForm.declared_credit_tier}
                    onChange={e => setManualForm({ ...manualForm, declared_credit_tier: e.target.value })}
                    style={{ width: '100%', padding: '6px', fontSize: '12px', border: '1px solid var(--border-medium)' }}
                  >
                    <option value="excellent">Excellent</option>
                    <option value="good">Good</option>
                    <option value="fair">Fair</option>
                    <option value="limited_history">Limited History</option>
                  </select>
                </div>
              </div>

              <div style={{ display: 'flex', justifyContent: 'flex-end', gap: '8px', marginTop: '16px' }}>
                <button type="button" className="btn" onClick={() => setManualModalOpen(false)}>
                  Cancel
                </button>
                <button type="submit" className="btn btn-primary" disabled={isScoringManual}>
                  {isScoringManual ? 'Evaluating...' : 'Score Application'}
                </button>
              </div>
            </form>
          </div>
        </div>
      )}

      {/* 6. Interactive JSON Upload Modal */}
      {jsonModalOpen && (
        <div style={{
          position: 'fixed',
          top: 0,
          left: 0,
          right: 0,
          bottom: 0,
          background: 'rgba(23, 25, 30, 0.45)',
          display: 'flex',
          alignItems: 'center',
          justifyContent: 'center',
          zIndex: 1000,
        }}>
          <div style={{
            background: 'var(--bg-surface)',
            width: '640px',
            maxHeight: '90vh',
            overflowY: 'auto',
            border: '1px solid var(--border-dark)',
            borderRadius: '4px',
            boxShadow: '0 10px 25px rgba(0,0,0,0.15)',
            padding: '24px',
          }}>
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: '16px' }}>
              <h2 className="font-serif" style={{ fontSize: '18px', fontWeight: 600 }}>
                Upload Insurance Applications (JSON)
              </h2>
              <button
                className="btn btn-sm"
                onClick={() => setJsonModalOpen(false)}
                style={{ border: 'none', background: 'transparent', fontSize: '16px', cursor: 'pointer' }}
              >
                ✕
              </button>
            </div>

            <p style={{ fontSize: '11px', color: 'var(--text-muted)', marginBottom: '14px' }}>
              Upload or paste a JSON array of applications (up to 100 records) or a single application object.
              The VAE engine scores each record and isolates validation errors safely.
            </p>

            {jsonError && (
              <div style={{
                padding: '8px 12px',
                background: '#fef2f2',
                border: '1px solid #fecaca',
                color: '#991b1b',
                fontSize: '11px',
                borderRadius: '2px',
                marginBottom: '14px',
                whiteSpace: 'pre-line',
              }}>
                {jsonError}
              </div>
            )}

            <form onSubmit={handleJsonUpload} style={{ display: 'flex', flexDirection: 'column', gap: '12px' }}>
              <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between' }}>
                <label style={{ fontSize: '11px', fontWeight: 600 }}>
                  Select File or Paste Payload
                </label>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <label className="btn btn-sm" style={{ cursor: 'pointer' }}>
                    Choose .json file
                    <input
                      type="file"
                      accept=".json"
                      onChange={handleFileSelect}
                      style={{ display: 'none' }}
                    />
                  </label>
                  <button
                    type="button"
                    className="btn btn-sm"
                    onClick={loadExampleJsonTemplate}
                  >
                    Insert Example JSON
                  </button>
                </div>
              </div>

              <textarea
                value={jsonInputText}
                onChange={e => { setJsonInputText(e.target.value); setJsonError(null); }}
                placeholder='[\n  {\n    "application_id": "APP-101",\n    "policy_type": "auto",\n    "coverage_amount": 50000,\n    "deductible_amount": 500,\n    "premium_frequency": "monthly",\n    "application_channel": "online_direct",\n    "applicant_age": 34,\n    "annual_income_estimate": 72000,\n    "prior_application_count": 0,\n    "recent_address_changes_count": 0,\n    "property_or_vehicle_age": 4,\n    "beneficiary_relation": "spouse",\n    "declared_credit_tier": "good"\n  }\n]'
                style={{
                  width: '100%',
                  height: '220px',
                  padding: '8px',
                  fontFamily: 'var(--font-mono)',
                  fontSize: '11px',
                  border: '1px solid var(--border-medium)',
                  borderRadius: '2px',
                  resize: 'vertical',
                  background: '#faf8f5',
                }}
              />

              <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginTop: '10px' }}>
                <span style={{ fontSize: '11px', color: 'var(--text-muted)' }}>
                  Synthetic educational demonstration. Safe per-record error formatting.
                </span>
                <div style={{ display: 'flex', gap: '8px' }}>
                  <button type="button" className="btn" onClick={() => setJsonModalOpen(false)}>
                    Cancel
                  </button>
                  <button type="submit" className="btn btn-primary" disabled={isUploadingJson}>
                    {isUploadingJson ? 'Scoring Batch...' : 'Score and Load'}
                  </button>
                </div>
              </div>
            </form>
          </div>
        </div>
      )}
    </div>
  );
}
