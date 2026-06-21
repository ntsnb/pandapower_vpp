type Props = {
  label: string;
  value: string | number;
  tone?: 'neutral' | 'good' | 'warn' | 'bad';
};

export function StatusCard({ label, value, tone = 'neutral' }: Props) {
  return (
    <section className={`stat-panel ${tone}`}>
      <span>{label}</span>
      <strong>{value}</strong>
    </section>
  );
}
