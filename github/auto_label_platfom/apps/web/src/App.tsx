import { Link, Navigate, Route, Routes, useLocation } from "react-router-dom";
import { clearAuth, getRole, getToken } from "./api";
import LoginPage from "./pages/LoginPage";
import ProjectsPage from "./pages/ProjectsPage";
import ProjectDetailPage from "./pages/ProjectDetailPage";
import StudioPage from "./pages/StudioPage";

function Shell({ children }: { children: React.ReactNode }) {
  const loc = useLocation();
  const role = getRole();
  return (
    <div className="layout">
      <aside className="nav">
        <h1>Auto Label Platform</h1>
        <p className="muted">机器人 / VLA · MVP</p>
        <Link className={loc.pathname.startsWith("/projects") ? "active" : ""} to="/projects">
          项目 Portal
        </Link>
        <div style={{ marginTop: "2rem" }}>
          <p className="muted">{role}</p>
          <button className="secondary" type="button" onClick={() => { clearAuth(); location.href = "/login"; }}>
            退出
          </button>
        </div>
      </aside>
      <main className="main">{children}</main>
    </div>
  );
}

function Private({ children }: { children: React.ReactNode }) {
  if (!getToken()) return <Navigate to="/login" replace />;
  return <Shell>{children}</Shell>;
}

export default function App() {
  return (
    <Routes>
      <Route path="/login" element={<LoginPage />} />
      <Route path="/" element={<Navigate to="/projects" replace />} />
      <Route path="/projects" element={<Private><ProjectsPage /></Private>} />
      <Route path="/projects/:id" element={<Private><ProjectDetailPage /></Private>} />
      <Route path="/studio/:jobId" element={<Private><StudioPage /></Private>} />
    </Routes>
  );
}
