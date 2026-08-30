import { useEffect, useState } from "react";
import { Link } from "react-router-dom";
import { api } from "../api";

type Project = {
  id: string;
  name: string;
  robot_type: string | null;
  fps: number;
  camera_keys: string[];
};

export default function ProjectsPage() {
  const [items, setItems] = useState<Project[]>([]);
  const [name, setName] = useState("demo-project");
  const [err, setErr] = useState("");

  async function load() {
    setItems(await api<Project[]>("/api/v1/projects"));
  }

  useEffect(() => {
    load().catch((e) => setErr(String(e)));
  }, []);

  async function create() {
    setErr("");
    try {
      await api("/api/v1/projects", { method: "POST", body: JSON.stringify({ name }) });
      await load();
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div>
      <h2>项目</h2>
      <div className="card row">
        <input value={name} onChange={(e) => setName(e.target.value)} placeholder="项目名" />
        <button type="button" onClick={create}>新建</button>
      </div>
      {err && <p className="error">{err}</p>}
      <div className="card">
        <table>
          <thead>
            <tr>
              <th>名称</th>
              <th>机器人</th>
              <th>fps</th>
              <th>相机</th>
            </tr>
          </thead>
          <tbody>
            {items.map((p) => (
              <tr key={p.id}>
                <td><Link to={`/projects/${p.id}`}>{p.name}</Link></td>
                <td>{p.robot_type || "—"}</td>
                <td>{p.fps}</td>
                <td className="muted">{(p.camera_keys || []).join(", ")}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
