import { FormEvent, useState } from "react";
import { useNavigate } from "react-router-dom";
import { api, setAuth } from "../api";

export default function LoginPage() {
  const nav = useNavigate();
  const [email, setEmail] = useState("manager@local");
  const [password, setPassword] = useState("manager123");
  const [err, setErr] = useState("");

  async function onSubmit(e: FormEvent) {
    e.preventDefault();
    setErr("");
    try {
      const data = await api<{ access_token: string; role: string }>("/api/v1/auth/login", {
        method: "POST",
        body: JSON.stringify({ email, password }),
      });
      setAuth(data.access_token, data.role);
      nav("/projects");
    } catch (ex) {
      setErr(ex instanceof Error ? ex.message : String(ex));
    }
  }

  return (
    <div className="login-page card">
      <h2>登录</h2>
      <p className="muted">seed：manager@local / manager123</p>
      <form onSubmit={onSubmit} className="row" style={{ flexDirection: "column", alignItems: "stretch" }}>
        <label>
          Email
          <input value={email} onChange={(e) => setEmail(e.target.value)} style={{ width: "100%" }} />
        </label>
        <label>
          Password
          <input type="password" value={password} onChange={(e) => setPassword(e.target.value)} style={{ width: "100%" }} />
        </label>
        {err && <p className="error">{err}</p>}
        <button type="submit">进入 Portal</button>
      </form>
    </div>
  );
}
