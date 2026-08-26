const defaultClassName = 'h-5 w-5'

function Icon({ children, className = defaultClassName, ...props }) {
  return <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round" className={className} aria-hidden="true" {...props}>{children}</svg>
}

export function ShieldIcon(props) { return <Icon {...props}><path d="M12 3 4.8 6v5.2c0 4.6 2.9 8.4 7.2 9.8 4.3-1.4 7.2-5.2 7.2-9.8V6L12 3Z" /><path d="m9.2 12 1.8 1.8 3.9-4" /></Icon> }
export function UploadIcon(props) { return <Icon {...props}><path d="M12 16V4" /><path d="m7.5 8.5 4.5-4.5 4.5 4.5" /><path d="M5 14.5v4A1.5 1.5 0 0 0 6.5 20h11a1.5 1.5 0 0 0 1.5-1.5v-4" /></Icon> }
export function ImageIcon(props) { return <Icon {...props}><rect x="3.5" y="4" width="17" height="16" rx="2" /><circle cx="9" cy="9.5" r="1.5" /><path d="m5.5 17 4.2-4.2 2.8 2.7 2.1-2.1 3.9 3.6" /></Icon> }
export function CheckIcon(props) { return <Icon {...props}><path d="m5 12.5 4.2 4.2L19 7" /></Icon> }
export function CloseIcon(props) { return <Icon {...props}><path d="m6 6 12 12M18 6 6 18" /></Icon> }
export function MenuIcon(props) { return <Icon {...props}><path d="M4 7h16M4 12h16M4 17h16" /></Icon> }
export function RefreshIcon(props) { return <Icon {...props}><path d="M20 7v5h-5" /><path d="M18.2 16a8 8 0 1 1 .7-8.8L20 12" /></Icon> }
export function TrashIcon(props) { return <Icon {...props}><path d="M4 7h16M9 3.5h6l1 3.5H8l1-3.5ZM6.5 7l.8 13h9.4l.8-13M10 11v5M14 11v5" /></Icon> }
export function SparkIcon(props) { return <Icon {...props}><path d="m12 3 1.4 4.1L17.5 8.5l-4.1 1.4L12 14l-1.4-4.1-4.1-1.4 4.1-1.4L12 3Z" /><path d="m18.5 14 .7 2.3 2.3.7-2.3.8-.7 2.2-.8-2.2-2.2-.8 2.2-.7.8-2.3Z" /></Icon> }
export function LockIcon(props) { return <Icon {...props}><rect x="5" y="10" width="14" height="10" rx="2" /><path d="M8.5 10V7.5a3.5 3.5 0 0 1 7 0V10M12 14v2" /></Icon> }
export function EyeOffIcon(props) { return <Icon {...props}><path d="m3 3 18 18" /><path d="M10.6 10.6a2 2 0 0 0 2.8 2.8M9.8 5.2A10.8 10.8 0 0 1 12 5c5.5 0 9 7 9 7a15.7 15.7 0 0 1-2.1 3M6.2 6.2C4.1 7.6 3 10 3 12c0 0 3.5 7 9 7 1.4 0 2.7-.5 3.8-1.2" /></Icon> }
export function FileSearchIcon(props) { return <Icon {...props}><path d="M13 3H6a2 2 0 0 0-2 2v14a2 2 0 0 0 2 2h6" /><path d="M13 3v5h5M13 3l5 5v3" /><circle cx="16" cy="16" r="3" /><path d="m18.2 18.2 2.3 2.3" /></Icon> }
export function SlidersIcon(props) { return <Icon {...props}><path d="M4 6h7M15 6h5M4 12h3M11 12h9M4 18h9M17 18h3" /><circle cx="13" cy="6" r="2" /><circle cx="9" cy="12" r="2" /><circle cx="15" cy="18" r="2" /></Icon> }
