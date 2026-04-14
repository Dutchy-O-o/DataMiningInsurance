// Theme toggle (persisted in localStorage)
function applyTheme(theme){
  if(theme==='light'){
    document.documentElement.setAttribute('data-theme','light');
    document.getElementById('themeToggle').innerHTML='&#9728;';
  } else {
    document.documentElement.removeAttribute('data-theme');
    document.getElementById('themeToggle').innerHTML='&#9790;';
  }
}
function toggleTheme(){
  const cur=document.documentElement.getAttribute('data-theme')==='light'?'dark':'light';
  localStorage.setItem('theme',cur);
  applyTheme(cur);
}
(function(){ const saved=localStorage.getItem('theme')||'dark'; applyTheme(saved); })();

let lastResult=null, lastData=null;

// Load welcome dashboard on page load
window.addEventListener('DOMContentLoaded', async () => {
  try {
    const res = await fetch('/api/stats');
    const data = await res.json();
    const ds = data.dataset;
    if (ds && ds.total_rows) {
      document.getElementById('statGrid').innerHTML = `
        <div class="stat-card blue"><div class="s-val">${ds.total_rows.toLocaleString()}</div><div class="s-label">Total Records</div></div>
        <div class="stat-card purple"><div class="s-val">$${ds.avg_cost.toLocaleString()}</div><div class="s-label">Avg Annual Cost</div></div>
        <div class="stat-card orange"><div class="s-val">${ds.smoker_pct}%</div><div class="s-label">Smokers</div></div>
        <div class="stat-card red"><div class="s-val">$${ds.smoker_avg.toLocaleString()}</div><div class="s-label">Avg Smoker Cost</div></div>
        <div class="stat-card green"><div class="s-val">$${ds.nonsmoker_avg.toLocaleString()}</div><div class="s-label">Avg Non-Smoker</div></div>
        <div class="stat-card cyan"><div class="s-val">${ds.avg_age}</div><div class="s-label">Avg Age</div></div>`;
    }
    if (data.models && data.models.length) {
      const mb = document.getElementById('modelBars');
      mb.innerHTML = '';
      data.models.forEach((m, i) => {
        const pct = (m.r2 / 1.0) * 100;
        const colors = ['linear-gradient(90deg,#3b82f6,#8b5cf6)','linear-gradient(90deg,#8b5cf6,#a78bfa)','linear-gradient(90deg,#06b6d4,#22d3ee)','linear-gradient(90deg,#64748b,#94a3b8)','linear-gradient(90deg,#64748b,#94a3b8)'];
        const best = m.is_best ? ' best' : '';
        const crown = m.is_best ? ' &#11088;' : '';
        mb.innerHTML += `<div class="model-row"><div class="model-name${best}">${m.name}${crown}</div><div class="model-bar-track"><div class="model-bar-fill" id="mbar${i}" style="background:${colors[i]};width:0;"></div></div><div class="model-r2">${(m.r2*100).toFixed(1)}%</div></div>`;
        setTimeout(()=>{document.getElementById('mbar'+i).style.width=pct+'%';}, 200+i*100);
      });
    }
  } catch(e) {}
});

function calcBmi(){const h=parseFloat(document.getElementById('height').value)/100;const w=parseFloat(document.getElementById('weight').value);if(h>0&&w>0)return w/(h*h);return 25;}
function updateBmi(){const bmi=calcBmi();let tag,cls;if(bmi<18.5){tag='Underweight';cls='underweight';}else if(bmi<25){tag='Normal';cls='normal';}else if(bmi<30){tag='Overweight';cls='overweight';}else{tag='Obese';cls='obese';}document.getElementById('bmiDisplay').innerHTML=`BMI: <strong>${bmi.toFixed(1)}</strong> <span class="bmi-tag ${cls}">${tag}</span>`;}

async function runPredict(){
  const btn=document.getElementById('predictBtn');
  btn.classList.add('loading');
  btn.innerHTML='<span class="spinner"></span>&nbsp; Analyzing...';
  const bmi=calcBmi();
  lastData={age:document.getElementById('age').value,sex:document.getElementById('sex').value,bmi:bmi.toFixed(1),children:document.getElementById('children').value,smoker:document.getElementById('smoker').value,region:document.getElementById('region').value};
  try{
    const res=await fetch('/predict',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify(lastData)});
    lastResult=await res.json();
    document.getElementById('welcome').style.display='none';
    const results=document.getElementById('results');results.classList.remove('show');void results.offsetWidth;results.classList.add('show');
    animateNumber('costAmount',lastResult.cost);

    // Confidence
    if(lastResult.confidence){
      const c=lastResult.confidence;
      document.getElementById('confLow').textContent='$'+c.low.toLocaleString();
      document.getElementById('confMid').textContent='$'+c.mid.toLocaleString();
      document.getElementById('confHigh').textContent='$'+c.high.toLocaleString();
      document.getElementById('confidence').style.display='flex';
    }

    // Scenarios
    const sc=document.getElementById('scenarios');sc.innerHTML='';
    if(lastResult.scenarios.quit_smoking){const s=lastResult.scenarios.quit_smoking;sc.innerHTML+=`<div class="scenario-card"><div class="sc-icon">&#128708;</div><div class="sc-title">If You Quit Smoking</div><div class="sc-new">$${s.new.toLocaleString()}</div><div class="sc-savings">&#9660; Save $${s.savings.toLocaleString()}/year</div></div>`;}
    if(lastResult.scenarios.healthy_bmi){const s=lastResult.scenarios.healthy_bmi;sc.innerHTML+=`<div class="scenario-card"><div class="sc-icon">&#127939;</div><div class="sc-title">If BMI Reaches 25.0</div><div class="sc-new">$${s.new.toLocaleString()}</div><div class="sc-savings">&#9660; Save $${s.savings.toLocaleString()}/year</div></div>`;}

    // SHAP
    if(lastResult.shap_bars&&lastResult.shap_bars.length){
      const container=document.getElementById('shapBars');container.innerHTML='';
      const maxAbs=Math.max(...lastResult.shap_bars.map(b=>Math.abs(b.value)));
      lastResult.shap_bars.forEach((bar,i)=>{
        const pct=(Math.abs(bar.value)/maxAbs)*48;const isPos=bar.value>=0;const cls=isPos?'positive':'negative';const sign=isPos?'+':'-';
        const row=document.createElement('div');row.className='shap-row';
        row.innerHTML=`<div class="shap-label">${bar.name}</div><div class="shap-bar-track"><div class="center-line"></div><div class="shap-bar-fill ${cls}" id="sf${i}" style="width:0%;"></div></div><div class="shap-val ${cls}">${sign}$${Math.abs(bar.value).toLocaleString(undefined,{maximumFractionDigits:0})}</div>`;
        container.appendChild(row);
        setTimeout(()=>{document.getElementById('sf'+i).style.width=pct+'%';},100+i*60);
      });
      document.getElementById('shapSection').style.display='block';
    }

    // AI Report
    const aiDiv=document.getElementById('aiReport');
    if(lastResult.advice){document.getElementById('reportBody').innerHTML=markdownToHtml(lastResult.advice);aiDiv.style.display='block';}else{aiDiv.style.display='none';}

    // Similar Patients
    const simDiv=document.getElementById('similarSection');
    if(lastResult.similar && lastResult.similar.patients && lastResult.similar.patients.length){
      const sim=lastResult.similar;
      const mkStat=(label,val,color)=>`<div style="background:linear-gradient(135deg,${color}15,${color}08);border:1px solid ${color}30;border-radius:14px;padding:16px 14px;text-align:center;"><div style="font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">${label}</div><div style="font-size:22px;font-weight:800;color:${color};letter-spacing:-0.5px;">$${val.toLocaleString()}</div></div>`;
      document.getElementById('similarSummary').innerHTML =
        mkStat('Lowest',sim.summary.min,'#10b981')+
        mkStat('Median',sim.summary.median,'#3b82f6')+
        mkStat('Average',sim.summary.mean,'#8b5cf6')+
        mkStat('Highest',sim.summary.max,'#f59e0b');

      let html = '<div style="display:flex;flex-direction:column;gap:8px;">';
      sim.patients.forEach((p,i) => {
        const isSmoker = p.smoker==='yes';
        const smokerBadge = isSmoker
          ? '<span style="background:rgba(239,68,68,0.12);color:#f87171;padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;">SMOKER</span>'
          : '<span style="background:rgba(16,185,129,0.12);color:#34d399;padding:3px 10px;border-radius:999px;font-size:10px;font-weight:700;letter-spacing:0.5px;">NON-SMOKER</span>';
        const simBarWidth = Math.min(100, p.similarity);
        const simBarColor = p.similarity > 90 ? '#10b981' : p.similarity > 70 ? '#3b82f6' : '#f59e0b';
        const regionCap = p.region.charAt(0).toUpperCase() + p.region.slice(1);
        const sexLetter = p.sex === 'male' ? 'M' : 'F';
        const sexColor = p.sex === 'male' ? '#60a5fa' : '#f472b6';

        html += `<div style="background:var(--surface2);border-radius:14px;padding:16px 18px;display:grid;grid-template-columns:40px 1fr auto;gap:16px;align-items:center;transition:all 0.2s;" onmouseover="this.style.background='rgba(139,92,246,0.06)';this.style.transform='translateY(-1px)';" onmouseout="this.style.background='';this.style.transform='';">
          <div style="width:36px;height:36px;border-radius:50%;background:linear-gradient(135deg,#8b5cf6,#a78bfa);display:flex;align-items:center;justify-content:center;font-weight:800;color:white;font-size:14px;box-shadow:0 4px 12px rgba(139,92,246,0.3);">${i+1}</div>
          <div>
            <div style="display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin-bottom:8px;">
              <span style="display:inline-flex;align-items:center;justify-content:center;width:22px;height:22px;border-radius:6px;background:${sexColor}22;color:${sexColor};font-size:11px;font-weight:800;">${sexLetter}</span>
              <span style="font-size:15px;color:var(--text);font-weight:600;">Age ${p.age}, BMI ${p.bmi}, ${regionCap}</span>
              ${smokerBadge}
            </div>
            <div style="display:flex;align-items:center;gap:8px;">
              <div style="flex:1;max-width:180px;height:6px;background:rgba(255,255,255,0.04);border-radius:3px;overflow:hidden;"><div style="width:${simBarWidth}%;height:100%;background:${simBarColor};border-radius:3px;transition:width 0.6s cubic-bezier(0.4,0,0.2,1);"></div></div>
              <span style="font-size:11px;color:var(--text3);font-weight:600;">${p.similarity}% match</span>
            </div>
          </div>
          <div style="text-align:right;">
            <div style="font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;margin-bottom:2px;">Actual Cost</div>
            <div style="font-size:22px;font-weight:800;color:${isSmoker?'#f87171':'#60a5fa'};letter-spacing:-0.5px;">$${p.actual_charge.toLocaleString()}</div>
          </div>
        </div>`;
      });
      html += '</div>';
      document.getElementById('similarList').innerHTML = html;
      simDiv.style.display='block';
    } else { simDiv.style.display='none'; }

    // Profile
    const sx=lastData.sex==='male'?'Male':'Female',sm=lastData.smoker==='yes'?'Yes':'No',rg=lastData.region.charAt(0).toUpperCase()+lastData.region.slice(1),h=document.getElementById('height').value,w=document.getElementById('weight').value;
    document.getElementById('profileBar').innerHTML=`<span>&#128100; Age <span class="pval">${lastData.age}</span></span><span>&#9878; Sex <span class="pval">${sx}</span></span><span>&#9878; ${h}cm/${w}kg <span class="pval">BMI ${parseFloat(lastData.bmi).toFixed(1)}</span></span><span>&#128118; Children <span class="pval">${lastData.children}</span></span><span>&#128684; Smoker <span class="pval">${sm}</span></span><span>&#127760; Region <span class="pval">${rg}</span></span>`;
  }catch(e){alert('Error: '+e.message);}
  btn.classList.remove('loading');btn.innerHTML='Calculate My Cost';
}

async function downloadReport(){if(!lastResult||!lastResult.advice)return;const res=await fetch('/download-report',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({cost:lastResult.cost,advice:lastResult.advice,age:lastData.age,bmi:lastData.bmi,smoker:lastData.smoker,region:lastData.region})});const blob=await res.blob();const url=URL.createObjectURL(blob);const a=document.createElement('a');a.href=url;a.download='insurance_report.md';a.click();URL.revokeObjectURL(url);}

function handleDrop(e){
  e.preventDefault();
  const dz=document.getElementById('dropZone');
  dz.classList.remove('drag-active');
  const files=e.dataTransfer.files;
  if(files.length){
    document.getElementById('csvFile').files=files;
    uploadBatch();
  }
}

async function uploadBatch(){
  const fileInput=document.getElementById('csvFile');
  const status=document.getElementById('batchStatus');
  const resultsDiv=document.getElementById('batchResults');
  if(!fileInput.files.length){return;}
  const file=fileInput.files[0];
  status.textContent='Uploading and processing '+file.name+'...';
  status.style.color='#3b82f6';
  resultsDiv.innerHTML='';

  const formData=new FormData();
  formData.append('file',file);

  try{
    const res=await fetch('/batch_predict',{method:'POST',body:formData});
    if(!res.ok){const err=await res.json();throw new Error(err.error||'Request failed');}
    const data=await res.json();

    status.textContent='Done! '+data.summary.successful+' predictions generated.';
    status.style.color='#10b981';

    const mkStat=(label,val,color,isCount)=>`<div style="background:linear-gradient(135deg,${color}15,${color}08);border:1px solid ${color}30;border-radius:14px;padding:16px 14px;text-align:center;"><div style="font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:1px;font-weight:600;margin-bottom:6px;">${label}</div><div style="font-size:22px;font-weight:800;color:${color};letter-spacing:-0.5px;">${isCount?val:('$'+val.toLocaleString())}</div></div>`;
    let html='<div style="display:grid;grid-template-columns:repeat(4,1fr);gap:12px;margin-bottom:20px;">';
    html+=mkStat('Total',data.summary.total,'#3b82f6',true);
    html+=mkStat('Success',data.summary.successful,'#10b981',true);
    html+=mkStat('Mean Cost',data.summary.mean_prediction,'#8b5cf6',false);
    html+=mkStat('Max Cost',data.summary.max_prediction,'#f59e0b',false);
    html+='</div>';

    html+='<div style="background:var(--surface2);border:1px solid var(--border);border-radius:14px;overflow:hidden;">';
    html+='<div style="display:grid;grid-template-columns:40px 60px 60px 60px 100px 1fr 140px;gap:10px;padding:12px 16px;background:rgba(59,130,246,0.05);border-bottom:1px solid var(--border);font-size:11px;color:var(--text3);text-transform:uppercase;letter-spacing:0.5px;font-weight:700;">';
    html+='<span>#</span><span>Age</span><span>Sex</span><span>BMI</span><span>Smoker</span><span>Top SHAP Feature</span><span style="text-align:right;">Predicted Cost</span></div>';
    html+='<div style="max-height:340px;overflow-y:auto;">';
    data.results.forEach((r,i) => {
      if(r.error){
        html+=`<div style="padding:10px 16px;color:#ef4444;font-size:12px;border-bottom:1px solid var(--border);">Row ${i+1} error: ${r.error}</div>`;
      }else{
        const isSmoker=r.smoker==='yes';
        const smokeTag=isSmoker
          ? '<span style="background:rgba(239,68,68,0.12);color:#f87171;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:700;">YES</span>'
          : '<span style="background:rgba(16,185,129,0.12);color:#34d399;padding:2px 8px;border-radius:6px;font-size:10px;font-weight:700;">NO</span>';
        html+=`<div style="display:grid;grid-template-columns:40px 60px 60px 60px 100px 1fr 140px;gap:10px;padding:10px 16px;border-bottom:1px solid rgba(255,255,255,0.03);font-size:13px;align-items:center;transition:background 0.15s;" onmouseover="this.style.background='rgba(59,130,246,0.04)';" onmouseout="this.style.background='';">
          <span style="color:var(--text3);font-weight:600;">${i+1}</span>
          <span style="color:var(--text);">${r.age}</span>
          <span style="color:var(--text);">${r.sex}</span>
          <span style="color:var(--text);">${r.bmi}</span>
          <span>${smokeTag}</span>
          <span style="color:var(--text2);font-size:12px;">${r.top_shap_feature}</span>
          <span style="text-align:right;font-weight:700;color:${isSmoker?'#f87171':'#60a5fa'};font-size:15px;">$${r.predicted_cost.toLocaleString()}</span>
        </div>`;
      }
    });
    html+='</div></div>';
    html+=`<button class="btn-download" style="background:linear-gradient(135deg,#06b6d4,#22d3ee);margin-top:16px;box-shadow:0 8px 20px rgba(6,182,212,0.25);color:white;border:none;font-weight:600;" onclick="downloadBatchCsv()">&#11015; Download Full Results (CSV)</button>`;
    resultsDiv.innerHTML=html;
    window._lastBatchFile=file;
  }catch(e){
    status.textContent='Error: '+e.message;
    status.style.color='#ef4444';
  }
}

async function downloadBatchCsv(){
  const file=window._lastBatchFile;
  if(!file){return;}
  const formData=new FormData();
  formData.append('file',file);
  const res=await fetch('/batch_predict?format=csv',{method:'POST',body:formData});
  const blob=await res.blob();
  const url=URL.createObjectURL(blob);
  const a=document.createElement('a');a.href=url;a.download='batch_predictions.csv';a.click();
  URL.revokeObjectURL(url);
}

function animateNumber(id,target){const el=document.getElementById(id);const duration=1000;const start=performance.now();function update(now){const p=Math.min((now-start)/duration,1);const ease=1-Math.pow(1-p,3);el.textContent='$'+Math.round(target*ease).toLocaleString();if(p<1)requestAnimationFrame(update);}requestAnimationFrame(update);}

function markdownToHtml(md){return md.replace(/### (.*)/g,'<h3>$1</h3>').replace(/## (.*)/g,'<h2>$1</h2>').replace(/# (.*)/g,'<h1>$1</h1>').replace(/\*\*(.*?)\*\*/g,'<strong>$1</strong>').replace(/\*(.*?)\*/g,'<em>$1</em>').replace(/^- (.*)/gm,'<li>$1</li>').replace(/(<li>.*<\/li>)/s,'<ul>$1</ul>').replace(/\n\n/g,'</p><p>').replace(/\n/g,'<br>');}