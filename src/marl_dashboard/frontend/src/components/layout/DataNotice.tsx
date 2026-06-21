type Props = {
  loading?: boolean;
  error?: string | null;
};

export function DataNotice({ loading = false, error = null }: Props) {
  if (error) {
    return <div className="notice error">{error}</div>;
  }
  if (loading) {
    return <div className="notice">Loading</div>;
  }
  return null;
}
