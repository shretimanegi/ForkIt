import { Anchor, Bell, CircleHelp } from 'lucide-react';

export default function Navbar() {
  return (
    <header className="navbar">
      <div className="brand"><span className="brand-mark"><Anchor size={18} /></span><span>SONAR<span className="brand-accent">WATCH</span></span></div>
      <div className="nav-context"><span>Marine debris detection</span><span className="nav-divider" /> Local analysis workspace</div>
      <div className="nav-actions"><button className="icon-button" aria-label="Help"><CircleHelp size={18} /></button><button className="icon-button" aria-label="Notifications"><Bell size={18} /></button><div className="avatar">B3</div></div>
    </header>
  );
}
