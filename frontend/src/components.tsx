import { AnimatePresence, motion } from "framer-motion";
import type { LucideIcon } from "lucide-react";
import {
  AlertTriangle,
  ArrowDownRight,
  ArrowUpRight,
  Check,
  ChevronDown,
  CircleHelp,
  Clock3,
  Database,
  ExternalLink,
  FileSearch,
  LayoutDashboard,
  Menu,
  Moon,
  Network,
  PanelLeftClose,
  Search,
  ShieldAlert,
  SlidersHorizontal,
  Sparkles,
  Sun,
  X,
  Zap,
} from "lucide-react";
import type { Factor, PageKey, RiskLevel, TransactionRow } from "./types";

export const navItems: { key: PageKey; label: string; icon: LucideIcon; detail: string }[] = [
  { key: "overview", label: "Overview", icon: LayoutDashboard, detail: "System pulse" },
  { key: "analyzer", label: "Transaction Analyzer", icon: Zap, detail: "Score a transaction" },
  { key: "investigation", label: "Fraud Investigation", icon: FileSearch, detail: "Review queue" },
  { key: "behavior", label: "Behavioral Intelligence", icon: Network, detail: "Signals & patterns" },
  { key: "performance", label: "Model Performance", icon: SlidersHorizontal, detail: "Evaluation lab" },
  { key: "explainable", label: "Explainable AI", icon: Sparkles, detail: "Model evidence" },
];

export function AppShell({
  page,
  setPage,
  children,
  theme,
  setTheme,
  mobileOpen,
  setMobileOpen,
}: {
  page: PageKey;
  setPage: (page: PageKey) => void;
  children: React.ReactNode;
  theme: "dark" | "light";
  setTheme: (theme: "dark" | "light") => void;
  mobileOpen: boolean;
  setMobileOpen: (open: boolean) => void;
}) {
  return (
    <div className="min-h-screen bg-ink text-slate-100">
      <Sidebar page={page} setPage={setPage} mobileOpen={mobileOpen} setMobileOpen={setMobileOpen} />
      <div className="lg:pl-[264px]">
        <Topbar page={page} setTheme={setTheme} theme={theme} setMobileOpen={setMobileOpen} />
        <main className="mx-auto max-w-[1600px] px-4 pb-12 pt-6 sm:px-6 lg:px-8">{children}</main>
      </div>
    </div>
  );
}

function Sidebar({ page, setPage, mobileOpen, setMobileOpen }: { page: PageKey; setPage: (page: PageKey) => void; mobileOpen: boolean; setMobileOpen: (open: boolean) => void }) {
  return (
    <>
      <AnimatePresence>
        {mobileOpen && <motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} onClick={() => setMobileOpen(false)} className="fixed inset-0 z-40 bg-black/70 lg:hidden" aria-label="Close navigation" />}
      </AnimatePresence>
      <aside className={`fixed inset-y-0 left-0 z-50 flex w-[264px] flex-col border-r border-line/70 bg-[#0d141e]/95 px-4 py-5 backdrop-blur-xl transition-transform duration-300 lg:translate-x-0 ${mobileOpen ? "translate-x-0" : "-translate-x-full"}`}>
        <div className="mb-8 flex items-center justify-between px-2">
          <button onClick={() => setPage("overview")} className="flex items-center gap-3 text-left">
            <span className="grid size-9 place-items-center rounded-xl bg-signal text-ink shadow-[0_0_24px_rgba(84,199,243,.35)]"><ShieldAlert size={20} strokeWidth={2.5} /></span>
            <span><span className="block font-display text-[17px] font-bold tracking-[-.03em]">BehaviorGuard</span><span className="block font-mono text-[9px] uppercase tracking-[.18em] text-slate-500">Fraud intelligence</span></span>
          </button>
          <button onClick={() => setMobileOpen(false)} className="icon-button lg:hidden" aria-label="Close navigation"><X size={18} /></button>
        </div>
        <div className="mb-3 px-2 font-mono text-[10px] uppercase tracking-[.18em] text-slate-600">Workspace</div>
        <nav className="space-y-1">
          {navItems.map((item) => {
            const Icon = item.icon;
            const active = page === item.key;
            return <button key={item.key} onClick={() => { setPage(item.key); setMobileOpen(false); }} className={`nav-item ${active ? "nav-item-active" : ""}`}><Icon size={17} /><span className="min-w-0 flex-1"><span className="block truncate text-[13px] font-medium">{item.label}</span><span className="block truncate text-[10px] text-slate-600">{item.detail}</span></span>{active && <span className="size-1.5 rounded-full bg-signal shadow-[0_0_10px_#54c7f3]" />}</button>;
          })}
        </nav>
        <div className="mt-auto space-y-3">
          <div className="rounded-2xl border border-signal/15 bg-signal/[.04] p-4">
            <div className="mb-3 flex items-center justify-between"><span className="font-mono text-[10px] uppercase tracking-[.16em] text-signal">Live model</span><span className="pulse-dot" /></div>
            <div className="font-display text-sm font-semibold">XGBoost final v1</div>
            <div className="mt-1 text-xs text-slate-500">Threshold locked at 0.54</div>
            <div className="mt-4 h-1 overflow-hidden rounded-full bg-slate-800"><div className="h-full w-[78%] rounded-full bg-signal" /></div>
            <div className="mt-2 flex justify-between font-mono text-[9px] text-slate-600"><span>MODEL HEALTH</span><span className="text-mint">OPERATIONAL</span></div>
          </div>
          <div className="flex items-center gap-3 border-t border-line/60 px-2 pt-4"><div className="grid size-8 place-items-center rounded-full bg-gradient-to-br from-slate-500 to-slate-700 text-xs font-bold">BG</div><div className="min-w-0"><div className="truncate text-xs font-medium text-slate-300">Analysis workspace</div><div className="truncate text-[10px] text-slate-600">Research environment</div></div><button className="ml-auto text-slate-600 hover:text-slate-300" title="Workspace options"><ChevronDown size={15} /></button></div>
        </div>
      </aside>
    </>
  );
}

function Topbar({ page, theme, setTheme, setMobileOpen }: { page: PageKey; theme: "dark" | "light"; setTheme: (theme: "dark" | "light") => void; setMobileOpen: (open: boolean) => void }) {
  const current = navItems.find((item) => item.key === page);
  return <header className="sticky top-0 z-30 border-b border-line/50 bg-ink/80 backdrop-blur-xl"><div className="flex h-[72px] items-center gap-4 px-4 sm:px-6 lg:px-8"><button className="icon-button lg:hidden" onClick={() => setMobileOpen(true)} aria-label="Open navigation"><Menu size={20} /></button><div className="min-w-0 flex-1"><div className="flex items-center gap-2 text-[11px] text-slate-600"><span>BehaviorGuard</span><span>/</span><span className="truncate text-slate-400">{current?.label}</span></div><h1 className="mt-0.5 truncate font-display text-lg font-semibold tracking-[-.03em] text-white">{current?.label}</h1></div><div className="hidden items-center gap-2 rounded-full border border-mint/15 bg-mint/[.04] px-3 py-1.5 sm:flex"><span className="pulse-dot bg-mint" /><span className="font-mono text-[10px] uppercase tracking-[.14em] text-mint">API connected</span></div><button className="icon-button" title="Search"><Search size={17} /></button><button className="icon-button" title="Help"><CircleHelp size={17} /></button><button className="icon-button" title="Toggle theme" onClick={() => setTheme(theme === "dark" ? "light" : "dark")}>{theme === "dark" ? <Sun size={17} /> : <Moon size={17} />}</button></div></header>;
}

export function PageHeader({ eyebrow, title, description, action }: { eyebrow: string; title: string; description: string; action?: React.ReactNode }) {
  return <div className="mb-7 flex flex-col justify-between gap-4 sm:flex-row sm:items-end"><div><div className="mb-2 flex items-center gap-2 font-mono text-[10px] uppercase tracking-[.18em] text-signal"><span className="size-1.5 rounded-full bg-signal" />{eyebrow}</div><h2 className="font-display text-3xl font-semibold tracking-[-.055em] text-white sm:text-4xl">{title}</h2><p className="mt-2 max-w-2xl text-sm leading-6 text-slate-500">{description}</p></div>{action}</div>;
}

export function Button({ children, variant = "primary", icon: Icon, onClick, type = "button" }: { children: React.ReactNode; variant?: "primary" | "secondary" | "ghost"; icon?: LucideIcon; onClick?: () => void; type?: "button" | "submit" }) {
  return <button type={type} onClick={onClick} className={`button button-${variant}`}>{Icon && <Icon size={15} />}{children}</button>;
}

export function KpiCard({ label, value, delta, detail, icon: Icon, tone = "cyan" }: { label: string; value: string; delta?: string; detail: string; icon: LucideIcon; tone?: "cyan" | "mint" | "amber" | "red" }) {
  return <motion.div whileHover={{ y: -2 }} className="panel relative overflow-hidden p-5"><div className={`absolute -right-7 -top-7 size-24 rounded-full blur-3xl ${tone === "red" ? "bg-danger/15" : tone === "amber" ? "bg-amber/15" : tone === "mint" ? "bg-mint/15" : "bg-signal/15"}`} /><div className="relative flex items-start justify-between"><div><div className="mb-4 font-mono text-[10px] uppercase tracking-[.16em] text-slate-500">{label}</div><div className="font-display text-3xl font-semibold tracking-[-.06em] text-white">{value}</div><div className="mt-2 flex items-center gap-2 text-xs"><span className={delta?.startsWith("-") ? "text-danger" : "text-mint"}>{delta}</span><span className="text-slate-600">{detail}</span></div></div><div className={`grid size-9 place-items-center rounded-xl border ${tone === "red" ? "border-danger/20 bg-danger/10 text-danger" : tone === "amber" ? "border-amber/20 bg-amber/10 text-amber" : tone === "mint" ? "border-mint/20 bg-mint/10 text-mint" : "border-signal/20 bg-signal/10 text-signal"}`}><Icon size={17} /></div></div></motion.div>;
}

export function Panel({ title, subtitle, children, action, className = "" }: { title?: string; subtitle?: string; children: React.ReactNode; action?: React.ReactNode; className?: string }) {
  return <section className={`panel ${className}`}><div className="flex items-start justify-between gap-4 border-b border-line/60 px-5 py-4"><div>{title && <h3 className="font-display text-sm font-semibold text-slate-200">{title}</h3>}{subtitle && <p className="mt-1 text-xs text-slate-600">{subtitle}</p>}</div>{action}</div><div className="p-5">{children}</div></section>;
}

export function RiskBadge({ risk }: { risk: RiskLevel }) {
  const config = { high: { label: "High risk", className: "badge-danger" }, medium: { label: "Review", className: "badge-amber" }, low: { label: "Low risk", className: "badge-mint" } }[risk];
  return <span className={`badge ${config.className}`}><span className="size-1.5 rounded-full bg-current" />{config.label}</span>;
}

export function FactorRow({ factor, rank }: { factor: Factor; rank?: number }) {
  const positive = factor.contribution > 0;
  return <div className="flex items-center gap-3 border-b border-line/40 py-3 last:border-0"><div className="grid size-7 shrink-0 place-items-center rounded-lg bg-slate-800 font-mono text-[10px] text-slate-500">{rank ?? ""}</div><div className="min-w-0 flex-1"><div className="flex items-center gap-2"><span className="truncate text-xs font-semibold text-slate-200">{factor.feature}</span><span className={`text-[10px] ${positive ? "text-danger" : "text-mint"}`}>{positive ? "increases risk" : "reduces risk"}</span></div><div className="mt-1 truncate font-mono text-[10px] text-slate-600">Observed: {factor.display_value}</div></div><div className="flex items-center gap-1 font-mono text-xs"><span className={positive ? "text-danger" : "text-mint"}>{positive ? <ArrowUpRight size={14} /> : <ArrowDownRight size={14} />}</span><span className={positive ? "text-danger" : "text-mint"}>{Math.abs(factor.contribution).toFixed(2)}</span></div></div>;
}

export function TransactionTable({ rows, onSelect }: { rows: TransactionRow[]; onSelect?: (row: TransactionRow) => void }) {
  return <div className="overflow-x-auto"><table className="data-table"><thead><tr><th>Transaction</th><th>Amount</th><th>Probability</th><th>Risk</th><th>Channel</th><th>Top signal</th><th>Status</th></tr></thead><tbody>{rows.map((row) => <tr key={row.id} onClick={() => onSelect?.(row)} className={onSelect ? "cursor-pointer" : ""}><td><div className="font-mono text-xs font-medium text-slate-200">{row.id}</div><div className="mt-1 text-[10px] text-slate-600">{row.time}</div></td><td className="font-mono text-xs text-slate-300">${row.amount.toLocaleString(undefined, { minimumFractionDigits: 2 })}</td><td><div className="flex items-center gap-2"><div className="h-1.5 w-16 overflow-hidden rounded-full bg-slate-800"><div className={`h-full rounded-full ${row.probability > .6 ? "bg-danger" : row.probability > .25 ? "bg-amber" : "bg-mint"}`} style={{ width: `${row.probability * 100}%` }} /></div><span className="font-mono text-xs text-slate-400">{Math.round(row.probability * 100)}%</span></div></td><td><RiskBadge risk={row.risk} /></td><td className="text-xs text-slate-500">{row.channel}</td><td><span className="rounded-md bg-slate-800 px-2 py-1 font-mono text-[10px] text-signal">{row.topSignal}</span></td><td>{row.reviewed ? <span className="inline-flex items-center gap-1 text-[11px] text-mint"><Check size={13} />Reviewed</span> : <span className="inline-flex items-center gap-1 text-[11px] text-amber"><Clock3 size={13} />Pending</span>}</td></tr>)}</tbody></table></div>;
}

export function Drawer({ open, onClose, title, children }: { open: boolean; onClose: () => void; title: string; children: React.ReactNode }) {
  return <AnimatePresence>{open && <><motion.button initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }} className="fixed inset-0 z-40 bg-black/70" onClick={onClose} aria-label="Close drawer" /><motion.aside initial={{ x: "100%" }} animate={{ x: 0 }} exit={{ x: "100%" }} transition={{ type: "spring", damping: 28, stiffness: 260 }} className="fixed right-0 top-0 z-50 h-full w-full max-w-lg overflow-y-auto border-l border-line bg-[#101824] shadow-2xl"><div className="flex items-center justify-between border-b border-line px-6 py-5"><div className="font-display font-semibold text-white">{title}</div><button className="icon-button" onClick={onClose} aria-label="Close drawer"><X size={18} /></button></div><div className="p-6">{children}</div></motion.aside></>}</AnimatePresence>;
}

export function ToastRegion({ message, onClose }: { message: string | null; onClose: () => void }) {
  return <AnimatePresence>{message && <motion.div initial={{ opacity: 0, y: -10 }} animate={{ opacity: 1, y: 0 }} exit={{ opacity: 0, y: -10 }} className="fixed right-5 top-5 z-[70] flex items-center gap-3 rounded-xl border border-mint/20 bg-[#14241f] px-4 py-3 text-xs text-mint shadow-2xl"><Check size={15} />{message}<button onClick={onClose} className="ml-2 text-mint/60 hover:text-mint"><X size={14} /></button></motion.div>}</AnimatePresence>;
}

export function EmptyState({ icon: Icon = Database, title, text }: { icon?: LucideIcon; title: string; text: string }) {
  return <div className="grid place-items-center py-14 text-center"><div className="mb-4 grid size-12 place-items-center rounded-2xl border border-line bg-slate-900 text-slate-600"><Icon size={21} /></div><h3 className="font-display text-sm font-semibold text-slate-300">{title}</h3><p className="mt-2 max-w-xs text-xs leading-5 text-slate-600">{text}</p></div>;
}

export function ErrorState({ text = "This view could not load right now." }: { text?: string }) {
  return <div className="flex items-center gap-3 rounded-xl border border-danger/20 bg-danger/5 p-4 text-xs text-danger"><AlertTriangle size={16} />{text}</div>;
}
