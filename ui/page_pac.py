import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import date, timedelta

from core.pac_core import (
    risk_label,
    expected_return_from_risk,
    PostPeriodRule,
    build_dca_schedule,
    compound_path,
    DCA_TEMPLATES,
    build_plan_from_template,
    allocation_table,
    parse_overrides,
    get_component_templates,
    build_performance_panel,
)


def _parse_sess_date(key: str, fallback: date) -> date:
    plan = st.session_state.get("dca_plan", {})
    v = plan.get(key)
    try:
        return pd.to_datetime(v).date() if v else fallback
    except Exception:
        return fallback


def render_pac_page() -> None:
    st.title("PAC & Investimenti programmati")
    st.caption(
        "Configura piani di accumulo (PAC / DCA), calcola interessi composti "
        "e analizza portafogli template ETF."
    )

    # ==========================================================================
    # 1) QUESTIONARIO PAC / DCA
    # ==========================================================================
    st.header("Configurazione del piano PAC / DCA")

    # 1) Frequenza
    st.subheader("1) Frequenza versamenti")
    freq = st.selectbox(
        "Seleziona la frequenza dei versamenti",
        options=["1 week", "2 weeks", "1 month", "2 months", "3 months", "6 months"],
        index=2,
        help="Intervallo tra un versamento e il successivo.",
    )

    # 2) Importo e modalità
    st.subheader("2) Importo e modalità")
    c1, c2 = st.columns([1, 1])
    with c1:
        amount = st.number_input(
            "Importo per versamento (€)",
            min_value=0.0,
            value=200.0,
            step=50.0,
            format="%.2f",
        )
    with c2:
        amount_type = st.radio("Tipo di importo", ["Fixed", "Variable"], horizontal=True)

    delta_pct = 0.0
    post_rule: PostPeriodRule | None = None
    enable_rule = False

    if amount_type == "Variable":
        delta_pct = st.number_input(
            "Variazione % base per periodo (può essere negativa)",
            min_value=-50.0,
            max_value=50.0,
            value=2.0,
            step=0.5,
            help="Esempio: +2% aumenta l'importo ogni periodo; -2% lo riduce gradualmente.",
        )

        st.markdown(
            "**Regola post-periodo (opzionale)** – applica una nuova regola dopo un certo punto nel tempo."
        )
        enable_rule = st.checkbox("Attiva regola post-periodo", value=False)

        if enable_rule:
            cR1, cR2 = st.columns([1, 1])
            with cR1:
                trigger_mode = st.selectbox(
                    "Quando applicare la regola?", ["After N deposits", "After a specific date"]
                )
            with cR2:
                action = st.selectbox(
                    "Azione",
                    ["Freeze amount", "Increase by % per period", "Decrease by % per period"],
                )

            N_val: int | None = None
            trg_date: date | None = None
            pct2: float | None = None

            if trigger_mode == "After N deposits":
                N_val = int(
                    st.number_input(
                        "Numero di versamenti dopo cui attivare la regola",
                        min_value=1,
                        value=12,
                        step=1,
                    )
                )
                trig_mode_clean = "N"
            else:
                trg_date = st.date_input("Data di attivazione (inclusa)")
                trig_mode_clean = "date"

            if action == "Freeze amount":
                act_clean = "freeze"
            elif action == "Increase by % per period":
                act_clean = "inc"
                pct2 = st.number_input(
                    "Nuova variazione % per periodo (post-trigger, >0)",
                    min_value=0.0,
                    max_value=50.0,
                    value=1.0,
                    step=0.5,
                )
            else:
                act_clean = "dec"
                pct2 = st.number_input(
                    "Nuova variazione % per periodo (post-trigger, >0)",
                    min_value=0.0,
                    max_value=50.0,
                    value=1.0,
                    step=0.5,
                )

            post_rule = PostPeriodRule(
                enabled=True,
                trigger_mode=trig_mode_clean,  # "N" o "date"
                N=N_val,
                date=trg_date,
                action=act_clean,  # "freeze" / "inc" / "dec"
                pct=pct2,
            )

    # 3) Rischio
    st.subheader("3) Esposizione al rischio")
    risk_level = st.slider(
        "Quanto rischio sei disposto a sopportare?",
        min_value=1,
        max_value=10,
        value=5,
        step=1,
        help="1 = molto basso … 10 = estremamente elevato",
    )
    st.caption(f"Livello selezionato: **{risk_level}/10 – {risk_label(risk_level)}**")

    # 4) Orizzonte temporale
    st.subheader("4) Orizzonte temporale del PAC")
    today = date.today()
    default_end = date(min(today.year + 3, 2100), today.month, min(today.day, 28))
    max_allowed = date(2100, 12, 31)
    min_allowed = date(1900, 1, 1)

    t1, t2 = st.columns(2)
    with t1:
        start_date = st.date_input(
            "Data di inizio",
            value=today,
            min_value=min_allowed,
            max_value=max_allowed,
        )
    with t2:
        end_date = st.date_input(
            "Data di fine",
            value=default_end,
            min_value=min_allowed,
            max_value=max_allowed,
        )

    # 5) Anteprima piano
    st.subheader("5) Anteprima piano e calendario versamenti")
    run_preview = st.button("📅 Genera calendario & anteprima importi")

    if run_preview:
        try:
            df = build_dca_schedule(
                freq=freq,
                amount=float(amount),
                amount_type=amount_type,  # "Fixed" o "Variable"
                delta_pct=float(delta_pct),
                start_date=start_date,
                end_date=end_date,
                rule=post_rule,
            )
        except ValueError as e:
            st.error(str(e))
            return

        if df.empty:
            st.warning("Nessuna data generata: controlla l'intervallo selezionato.")
            return

        total_contrib = float(df["Deposit (€)"].sum())
        n_contrib = int(len(df))
        avg_contrib = float(df["Deposit (€)"].mean())

        k1, k2, k3 = st.columns(3)
        k1.metric("Numero versamenti", f"{n_contrib}")
        k2.metric("Capitale versato", f"€ {total_contrib:,.2f}")
        k3.metric("Importo medio", f"€ {avg_contrib:,.2f}")

        st.dataframe(df, use_container_width=True)
        st.caption("Anteprima completa del piano generato.")

        # ------------------- Calcolatore interesse composto -------------------
        st.markdown("### 📈 Calcolatore interesse composto (stima)")
        default_er = expected_return_from_risk(int(risk_level))
        er_pct = (
            st.slider(
                "Rendimento annuo atteso (%)",
                min_value=0.0,
                max_value=20.0,
                value=round(default_er * 100, 2),
                step=0.10,
                help="Usato per stimare la crescita tra un versamento e l'altro.",
            )
            / 100.0
        )

        comp_df = compound_path(
            dates_list=pd.to_datetime(df["Date"]),
            amounts_list=df["Deposit (€)"],
            end_dt=end_date,
            er_annual=er_pct,
        )

        if comp_df.empty:
            st.warning("Impossibile calcolare il percorso composto.")
        else:
            last = comp_df.iloc[-1]
            fv = float(last["Estimated value (€)"])
            cum = float(last["Cumulative contributions (€)"])
            intr = float(last["Estimated interest (€)"])
            roi = ((fv / cum) - 1.0) if cum > 0 else 0.0

            c1, c2, c3 = st.columns(3)
            c1.metric("Valore finale stimato", f"€ {fv:,.2f}")
            c2.metric("Interessi stimati", f"€ {intr:,.2f}")
            c3.metric("ROI sul capitale versato", f"{roi * 100:,.2f}%")

            fig, ax = plt.subplots(figsize=(12, 5))
            ax.plot(
                pd.to_datetime(comp_df["Date"]),
                comp_df["Estimated value (€)"],
                label="Valore stimato",
                linewidth=2.2,
            )
            ax.plot(
                pd.to_datetime(comp_df["Date"]),
                comp_df["Cumulative contributions (€)"],
                label="Capitale versato cumulato",
                linewidth=1.8,
            )
            ax.set_title("Evoluzione PAC – valore stimato vs capitale versato")
            ax.set_ylabel("€")
            ax.grid(True, alpha=0.3)
            ax.legend(loc="best")
            fig.tight_layout()
            st.pyplot(fig)

            with st.expander(
                "📄 Tabella dettagliata interesse composto & export", expanded=False
            ):
                st.dataframe(comp_df, use_container_width=True)
                csv_bytes2 = comp_df.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Scarica risultati composti (CSV)",
                    data=csv_bytes2,
                    file_name="dca_compound.csv",
                    mime="text/csv",
                )

        # Download piano
        csv_bytes = df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "⬇️ Scarica piano PAC (CSV)",
            data=csv_bytes,
            file_name="dca_schedule.csv",
            mime="text/csv",
        )

        # Salvataggio in sessione (per template veloci)
        st.session_state["dca_plan"] = {
            "frequency": freq,
            "base_amount": float(amount),
            "amount_type": amount_type,
            "base_delta_pct": float(delta_pct),
            "risk_level": int(risk_level),
            "risk_label": risk_label(risk_level),
            "start_date": str(start_date),
            "end_date": str(end_date),
        }

        with st.expander("🔧 Dettagli tecnici & note", expanded=False):
            st.markdown(
                "- Le frequenze **mensili** cercano di preservare lo stesso **giorno del mese**; "
                "se non esiste in un dato mese (es. 31 → febbraio) viene usato l'ultimo giorno valido.\n"
                "- Con importo **variabile**, la variazione base viene **capitalizzata per periodo**; "
                "la **regola post-periodo** sostituisce la dinamica dal momento del trigger in poi.\n"
                "- Il calcolatore di interesse composto usa una capitalizzazione **giornaliera** "
                "fino alla **data di fine**.\n"
            )

    # ==========================================================================
    # 2) TEMPLATE PAC (versamenti)
    # ==========================================================================
    with st.expander("📚 Template PAC – strategie pronte (versamenti)", expanded=False):
        st.caption(
            "Seleziona un template (es. DCA mensile fisso) e genera un piano veloce "
            "senza toccare il questionario principale."
        )
        if not DCA_TEMPLATES:
            st.info("Nessun template definito in core.pac_core.DCA_TEMPLATES.")
        else:
            names = [t["name"] for t in DCA_TEMPLATES]
            tmpl_sel = st.selectbox("Template", options=names, index=0)
            tmpl = next(t for t in DCA_TEMPLATES if t["name"] == tmpl_sel)
            st.caption(tmpl.get("desc", ""))

            today = date.today()
            default_tmpl_start = _parse_sess_date("start_date", today)
            default_tmpl_end = _parse_sess_date(
                "end_date",
                date(min(today.year + 3, 2100), today.month, min(today.day, 28)),
            )

            cD1, cD2 = st.columns(2)
            with cD1:
                tmpl_start = st.date_input(
                    "Data inizio (template)",
                    value=default_tmpl_start,
                    key="pac_tmpl_start",
                )
            with cD2:
                tmpl_end = st.date_input(
                    "Data fine (template)",
                    value=default_tmpl_end,
                    key="pac_tmpl_end",
                )

            cP1, cP2, cP3, cP4 = st.columns([1, 1, 1, 1.6])
            with cP1:
                st.write("**Frequenza**")
                st.code(tmpl["freq"])
            with cP2:
                st.write("**Importo (€)**")
                st.code(f'{tmpl["amount"]:.2f}')
            with cP3:
                st.write("**Tipo**")
                st.code(tmpl["type"])
            with cP4:
                if tmpl["type"] == "Variable":
                    rule = tmpl.get("rule", {}) or {}
                    if rule.get("enabled"):
                        if rule.get("trigger_mode") == "N":
                            when = f"after {rule.get('N')} deposits"
                        else:
                            when = f"from {rule.get('date', 'date')}"
                        if rule["action"] == "freeze":
                            rule_txt = f"Freeze {when}"
                        else:
                            sign = "+" if rule["action"] == "inc" else "-"
                            rule_txt = f"{sign}{rule.get('pct', 0)}%/period {when}"
                    else:
                        rule_txt = "—"
                    st.write("**Δ base & regola**")
                    st.code(f"Δ base: {tmpl['delta_base']:+.2f}% • Regola: {rule_txt}")
                else:
                    st.write("**Δ base & regola**")
                    st.code("N/A (importo fisso)")

            if st.button("🧪 Genera anteprima dal template", key="pac_tmpl_run"):
                if tmpl_end <= tmpl_start:
                    st.error("La **data fine** deve essere successiva alla **data inizio**.")
                else:
                    df_t = build_plan_from_template(tmpl, tmpl_start, tmpl_end)
                    if df_t.empty:
                        st.warning("Il template non ha generato alcuna data nel range selezionato.")
                    else:
                        tot_t = float(df_t["Deposit (€)"].sum())
                        n_t = int(len(df_t))
                        med_t = float(df_t["Deposit (€)"].mean())

                        k1, k2, k3 = st.columns(3)
                        k1.metric("Numero versamenti (template)", f"{n_t}")
                        k2.metric("Capitale versato (template)", f"€ {tot_t:,.2f}")
                        k3.metric("Importo medio (template)", f"€ {med_t:,.2f}")

                        st.dataframe(df_t, use_container_width=True)
                        csv_bytes = df_t.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            "⬇️ Scarica template (CSV)",
                            data=csv_bytes,
                            file_name="dca_template.csv",
                            mime="text/csv",
                            key="pac_tmpl_dl",
                        )

            with st.expander("ℹ️ Guida rapida ai template", expanded=False):
                st.markdown(
                    "- **Fixed Monthly / Micro-DCA / Bi-weekly**: regolarità e disciplina; "
                    "aiutano a ridurre il rischio di timing.\n"
                    "- **Gradual Growth / Aggressive Growth**: per reddito in crescita o orizzonti lunghi; "
                    "aumentano l'impegno nel tempo.\n"
                    "- **Glidepath Decreasing**: utile quando ti avvicini a un obiettivo (es. pensione), "
                    "riduce gradualmente l'esposizione.\n"
                    "- **Step-up / Freeze dopo N**: per programmare aumenti o stabilizzazioni dopo una fase "
                    "iniziale di accumulo.\n"
                )

    # ==========================================================================
    # 3) TEMPLATE COMPONENTI (PORTAFOGLI ETF)
    # ==========================================================================
    st.markdown("---")
    with st.expander("🧩 Template componenti – Portafogli ETF (US / UCITS)", expanded=False):
        st.caption(
            "Scegli una struttura di portafoglio (es. 60/40, All Weather), visualizza i pesi, "
            "la ripartizione per asset class e la performance normalizzata (Base=100). "
            "I dati di prezzo vengono caricati tramite Marketstack."
        )

        universe = st.radio(
            "Universo ETF",
            ["US ETFs (US-domiciled)", "UCITS / EU ETFs"],
            index=0,
            horizontal=True,
        )

        templates = get_component_templates(universe)
        tpl_name = st.selectbox("Template di portafoglio", options=list(templates.keys()))
        capital = st.number_input(
            "Capitale complessivo (EUR)",
            min_value=0.0,
            value=10_000.0,
            step=500.0,
            format="%.2f",
        )
        meta = templates[tpl_name]
        components = meta["items"]
        st.caption(meta["desc"])

        df_alloc = allocation_table(components, capital)

        c1, c2 = st.columns([1, 1])
        with c1:
            st.markdown("#### Componenti & pesi")
            if df_alloc.empty:
                st.info("Nessuna componente definita per il template selezionato.")
            else:
                st.dataframe(df_alloc, use_container_width=True)
                csv_bytes = df_alloc.to_csv(index=False).encode("utf-8")
                st.download_button(
                    "⬇️ Scarica allocazione (CSV)",
                    data=csv_bytes,
                    file_name="template_allocation.csv",
                    mime="text/csv",
                )

        with c2:
            if not df_alloc.empty:
                fig, ax = plt.subplots(figsize=(5.4, 5.4))
                grp = df_alloc.groupby("asset_class", as_index=False)["weight"].sum()
                ax.pie(
                    grp["weight"].values,
                    labels=grp["asset_class"].values,
                    autopct="%1.1f%%",
                    startangle=90,
                )
                ax.axis("equal")
                st.pyplot(fig)

        st.markdown("---")
        st.markdown("### 📊 Performance normalizzata (Base=100)")

        colA, colB, colC = st.columns([1, 1, 1.2])
        with colA:
            start_d = st.date_input("Da", value=(date.today() - timedelta(days=365)))
        with colB:
            end_d = st.date_input("A", value=date.today())
        with colC:
            freq_label = st.selectbox(
                "Frequenza di campionamento",
                ["Daily (Business)", "Weekly (Friday)", "Monthly"],
                index=0,
            )

        if "US ETFs" in universe:
            suggested = ["SPY", "VT", "AGG", "GLD", "DBC"]
        else:
            suggested = ["CSPX", "VWCE", "AGGH", "SGLD", "CMOD"]

        bm = st.multiselect("Benchmark", options=suggested, default=[suggested[0]])
        extra = st.text_input(
            "Ticker extra (opzionale, separati da virgola o spazio)",
            value="",
        )
        overrides_txt = st.text_input(
            "Override ticker (opzionale)",
            value="",
            help="Formato: TICK1=MS1; TICK2=MS2 (es. VWCE=VWCE.XETRA).",
        )

        show_template = st.checkbox("Mostra linea portafoglio (pesi template)", value=True)
        show_components = st.checkbox("Mostra singole componenti", value=True)
        show_bench = st.checkbox("Mostra benchmark", value=True)

        go = st.button("▶️ Calcola performance")

        if go:
            if end_d <= start_d:
                st.error("La **data fine** deve essere successiva alla **data inizio**.")
            elif df_alloc.empty:
                st.error("Nessuna componente valida nel template.")
            else:
                tokens = [t.strip() for t in extra.replace(",", " ").split() if t.strip()]
                bench_syms = list(dict.fromkeys((bm or []) + tokens))  # de-dup
                overrides_map = parse_overrides(overrides_txt)

                panel, comp_df_sym, no_data = build_performance_panel(
                    comp_df=df_alloc,
                    bench_syms=bench_syms,
                    overrides_map=overrides_map,
                    start_d=start_d,
                    end_d=end_d,
                    freq_label=freq_label,
                )

                if panel.empty:
                    st.error(
                        "Impossibile recuperare i dati di prezzo per i ticker selezionati. "
                        "Controlla i simboli o gli override Marketstack."
                    )
                else:
                    if no_data:
                        st.info("Nessuna serie trovata per: " + ", ".join(sorted(set(no_data))))

                    port_line = None
                    valid_rows = comp_df_sym[comp_df_sym["ms_symbol"].isin(panel.columns)]
                    if not valid_rows.empty:
                        w = (valid_rows["weight"] / valid_rows["weight"].sum()).values
                        M = panel[valid_rows["ms_symbol"]].values  # già Base=100
                        port_np = np.nansum(M * w, axis=1)
                        port_line = pd.Series(port_np, index=panel.index, name="Template (pesi)")

                    fig, ax = plt.subplots(figsize=(12, 6))

                    if show_template and port_line is not None and not port_line.dropna().empty:
                        ax.plot(
                            port_line.index,
                            port_line.values,
                            label="Template (pesi ponderati)",
                            linewidth=2.8,
                        )

                    if show_components:
                        for _, r in valid_rows.iterrows():
                            sym = r["ms_symbol"]
                            if isinstance(sym, str) and sym in panel.columns:
                                ax.plot(
                                    panel.index,
                                    panel[sym].values,
                                    linewidth=1.3,
                                    label=str(r["ticker"]),
                                )

                    if show_bench:
                        for sym in bench_syms:
                            if sym in panel.columns:
                                ax.plot(
                                    panel.index,
                                    panel[sym].values,
                                    linewidth=1.8,
                                    label=str(sym),
                                )

                    ax.set_title(
                        f"Performance normalizzata (Base=100) — "
                        f"{pd.to_datetime(start_d).date()} → {pd.to_datetime(end_d).date()}"
                    )
                    ax.set_ylabel("Indice (Base=100)")
                    ax.grid(True, alpha=0.3)
                    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
                    fig.tight_layout()
                    st.pyplot(fig)

                    with st.expander("📄 Dati & export", expanded=False):
                        st.dataframe(panel, use_container_width=True)
                        csv_bytes = panel.to_csv(index=True).encode("utf-8")
                        st.download_button(
                            "⬇️ Scarica serie Base=100 (CSV)",
                            data=csv_bytes,
                            file_name="template_benchmark_base100.csv",
                            mime="text/csv",
                        )
