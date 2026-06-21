import 'katex/dist/katex.min.css';

import { BlockMath } from 'react-katex';

import type { MetricRow } from '../../api/types';
import type { MetricHierarchyScope } from '../../utils/metricHierarchy';
import { sortFormulaEntriesByHierarchy } from '../../utils/metricHierarchy';
import { formulaSectionForMetric, formulaSections } from '../../utils/rewardCostSemantics';

type Props = {
  formulas: Record<string, string>;
  rows?: MetricRow[];
  scope?: MetricHierarchyScope;
  title?: string;
};

type FormulaMetadata = Pick<MetricRow, 'display_name' | 'description' | 'unit'>;

function metadataByMetricName(rows: MetricRow[]): Map<string, FormulaMetadata> {
  const metadata = new Map<string, FormulaMetadata>();
  for (const row of rows) {
    const existing = metadata.get(row.metric_name);
    const candidate = {
      display_name: row.display_name ?? existing?.display_name ?? null,
      description: row.description ?? existing?.description ?? null,
      unit: row.unit ?? existing?.unit ?? null
    };
    if (!existing || candidate.display_name || candidate.description || candidate.unit) {
      metadata.set(row.metric_name, candidate);
    }
  }
  return metadata;
}

export function FormulaTable({ formulas, rows = [], scope = 'default', title = '公式字典 / Formula Dictionary' }: Props) {
  const entries = sortFormulaEntriesByHierarchy(Object.entries(formulas), scope);
  const metadata = metadataByMetricName(rows);
  const entriesBySection = formulaSections.map((section) => ({
    ...section,
    entries: entries.filter(([name]) => formulaSectionForMetric(name, scope) === section.id)
  }));
  return (
    <section className="panel table-panel">
      <div className="panel-header">
        <h2>{title}</h2>
        <span>{entries.length} 个公式 / formulas</span>
      </div>
      <div className="formula-list">
        {entriesBySection.map(({ id, label, entries: sectionEntries }) =>
          sectionEntries.length > 0 ? (
            <div key={id} className="formula-section">
              <h3 className="formula-section-title">{label}</h3>
              {sectionEntries.map(([name, formula]) => {
                const rowMetadata = metadata.get(name);
                const displayName = rowMetadata?.display_name || name;
                return (
                  <article key={name} className="formula-row">
                    <div className="formula-metadata">
                      <h3 className="formula-title">{displayName}</h3>
                      {displayName !== name ? <code className="formula-code">{name}</code> : null}
                      {rowMetadata?.unit ? <span className="formula-unit">单位 / Unit: {rowMetadata.unit}</span> : null}
                      {rowMetadata?.description ? (
                        <p className="formula-description">{rowMetadata.description}</p>
                      ) : null}
                    </div>
                    <BlockMath math={formula} />
                  </article>
                );
              })}
            </div>
          ) : null
        )}
        {entries.length === 0 ? <div className="empty-state">无公式 / No formulas</div> : null}
      </div>
    </section>
  );
}
