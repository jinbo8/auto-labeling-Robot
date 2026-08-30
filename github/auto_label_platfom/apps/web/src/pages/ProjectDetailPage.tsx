import { useEffect, useState } from "react";
import { Link, useParams } from "react-router-dom";
import { api } from "../api";

type Episode = {
  episode_index: number;
  length: number;
  duration_s: number;
  task_text: string | null;
};

type Job = {
  id: string;
  episode_index: number;
  status: string;
  assignee_id: string | null;
};

export default function ProjectDetailPage() {
  const { id } = useParams();
  const [source, setSource] = useState(
    "/home/jin/6t/item/auto-labeling-Robot/lerobot/datasets/svla_so100_pickplace"
  );
  const [episodes, setEpisodes] = useState<Episode[]>([]);
  const [total, setTotal] = useState(0);
  const [jobs, setJobs] = useState<Job[]>([]);
  const [qa, setQa] = useState<any>(null);
  const [exportInfo, setExportInfo] = useState<any>(null);
  const [msg, setMsg] = useState("");
  const [err, setErr] = useState("");

  async function refresh() {
    const ep = await api<{ total: number; items: Episode[] }>(`/api/v1/projects/${id}/episodes?limit=20`);
    setEpisodes(ep.items);
    setTotal(ep.total);
    setJobs(await api<Job[]>(`/api/v1/jobs?project_id=${id}`));
  }

  useEffect(() => {
    refresh().catch((e) => setErr(String(e)));
  }, [id]);

  async function doImport() {
    setErr("");
    setMsg("导入中…");
    try {
      const imp = await api<any>(`/api/v1/projects/${id}/imports`, {
        method: "POST",
        body: JSON.stringify({ source_uri: source, format: "lerobot_v3" }),
      });
      setMsg(`导入完成 status=${imp.status} episodes≈${imp.meta_snapshot?.total_episodes}`);
      await refresh();
    } catch (e) {
      setErr(String(e));
      setMsg("");
    }
  }

  async function doQa() {
    setErr("");
    try {
      const r = await api(`/api/v1/projects/${id}/qa`, {
        method: "POST",
        body: JSON.stringify({ align_only: true }),
      });
      setQa(r);
      setMsg("QA 完成");
    } catch (e) {
      setErr(String(e));
    }
  }

  async function splitJobs() {
    setErr("");
    try {
      const r = await api<{ created: number }>(`/api/v1/projects/${id}/jobs/split`, {
        method: "POST",
        body: JSON.stringify({ episode_from: 0, episode_to: 9 }),
      });
      setMsg(`新建 Job ${r.created} 个`);
      await refresh();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function prelabel() {
    setErr("");
    try {
      const r = await api<any>(`/api/v1/projects/${id}/prelabel`, {
        method: "POST",
        body: JSON.stringify({ strategy: "sam3_text_keyframes", frame_stride: 30 }),
      });
      setMsg(`预标 predictions=${r.created_predictions}`);
      await refresh();
    } catch (e) {
      setErr(String(e));
    }
  }

  async function doExport() {
    setErr("");
    try {
      const r = await api(`/api/v1/projects/${id}/exports`, {
        method: "POST",
        body: JSON.stringify({ formats: ["coco", "lerobot_sidecar"] }),
      });
      setExportInfo(r);
      setMsg("导出完成");
    } catch (e) {
      setErr(String(e));
    }
  }

  return (
    <div>
      <h2>项目详情</h2>
      <p className="muted">id={id} · episodes={total}</p>

      <div className="card">
        <h3>导入 LeRobot v3</h3>
        <div className="row">
          <input style={{ flex: 1, minWidth: 280 }} value={source} onChange={(e) => setSource(e.target.value)} />
          <button type="button" onClick={doImport}>导入</button>
          <button type="button" className="secondary" onClick={doQa}>跑 QA</button>
          <button type="button" className="secondary" onClick={splitJobs}>拆 Job 0–9</button>
          <button type="button" className="secondary" onClick={prelabel}>批量预标</button>
          <button type="button" className="secondary" onClick={doExport}>导出</button>
        </div>
        {msg && <p className="muted">{msg}</p>}
        {err && <p className="error">{err}</p>}
      </div>

      {qa && (
        <div className="card">
          <h3>QA</h3>
          <pre className="muted" style={{ whiteSpace: "pre-wrap" }}>{JSON.stringify(qa.summary, null, 2)}</pre>
        </div>
      )}
      {exportInfo && (
        <div className="card">
          <h3>导出</h3>
          <p className="muted">{exportInfo.artifact_uri}</p>
        </div>
      )}

      <div className="card">
        <h3>Episodes（前 20）</h3>
        <table>
          <thead>
            <tr><th>#</th><th>帧数</th><th>时长</th><th>task</th></tr>
          </thead>
          <tbody>
            {episodes.map((e) => (
              <tr key={e.episode_index}>
                <td>{e.episode_index}</td>
                <td>{e.length}</td>
                <td>{e.duration_s.toFixed(1)}s</td>
                <td className="muted">{e.task_text}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>

      <div className="card">
        <h3>Jobs</h3>
        <table>
          <thead>
            <tr><th>episode</th><th>status</th><th>Studio</th></tr>
          </thead>
          <tbody>
            {jobs.map((j) => (
              <tr key={j.id}>
                <td>{j.episode_index}</td>
                <td>{j.status}</td>
                <td><Link to={`/studio/${j.id}`}>打开</Link></td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
