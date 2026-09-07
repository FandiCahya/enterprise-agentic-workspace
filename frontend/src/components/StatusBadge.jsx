export function StatusBadge({ status }) {
  const config = {
    online:   { label: 'API Online',   cls: 'online' },
    offline:  { label: 'API Offline',  cls: 'offline' },
    checking: { label: 'Checking...',  cls: 'checking' },
  };
  const { label, cls } = config[status] ?? config.checking;

  return (
    <span className={`status-badge ${cls}`}>
      <span className="dot" />
      {label}
    </span>
  );
}
