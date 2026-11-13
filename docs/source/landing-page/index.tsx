// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

import SiteLink, { LinkIndicator } from "./_links";
import CodeBlock from "@theme/CodeBlock";
import Layout from "@theme/Layout";
import type { ReactElement } from "react";

import {
  capabilities,
  discoveryExample,
  plugins,
  title,
  headline,
  headlineSecond,
  lead,
} from "./_homepage";
import styles from "./index.module.css";

export default function Home(): ReactElement {
  return (
    <Layout title={title} description={lead}>
      <main className={styles.surface}>
        <div className={styles.page}>
          <section className={styles.hero} aria-labelledby="landing-title">
            <div className={styles.heroCopy}>
              <h1 id="landing-title">{headline}</h1>
              <p className={styles.heroTagline}>{headlineSecond}</p>
              <p className={styles.lead}>{lead}</p>
              <div className={styles.actions}>
                <SiteLink
                  className={styles.primaryButton}
                  to="/documentation#get-started"
                >
                  Get started
                </SiteLink>
                <SiteLink
                  className={styles.secondaryButton}
                  to="/documentation"
                >
                  Browse documentation
                </SiteLink>
              </div>
              <SiteLink
                className={styles.repositoryLink}
                to="https://github.com/arm/mlia"
              >
                View on GitHub
              </SiteLink>
            </div>
          </section>

          <section
            className={styles.analysisSection}
            aria-labelledby="analysis-title"
          >
            <div className={styles.analysisHeader}>
              <p className={styles.eyebrow}>Inside the analysis</p>
              <h2 id="analysis-title">One model. Three perspectives.</h2>
            </div>
            <div className={styles.analysisBody}>
              <div className={styles.heroVisual}>
                <div className={styles.visualLabel}>
                  <span /> MLIA / COMMAND LINE
                </div>
                <div className={styles.analysisConsole}>
                  <p className={styles.consoleTitle}>
                    Discover your environment
                  </p>
                  <CodeBlock language="bash">{discoveryExample}</CodeBlock>
                  <p className={styles.consoleNote}>
                    Available checks depend on your installed plugins and
                    backends.
                  </p>
                </div>
              </div>
              <div
                className={styles.capabilities}
                aria-label="What you can do with MLIA"
              >
                {capabilities.map((item) => (
                  <SiteLink
                    className={styles.capabilityCard}
                    indicator={false}
                    to={item.to}
                    key={item.number}
                  >
                    <span className={styles.number}>{item.number}</span>
                    <h2>{item.title}</h2>
                    <p>{item.description}</p>
                    <span className={styles.cardLink}>
                      {item.link} <LinkIndicator to={item.to} />
                    </span>
                  </SiteLink>
                ))}
              </div>
            </div>
          </section>

          <div className={styles.workspace}>
            <section
              className={styles.startSection}
              aria-labelledby="start-title"
            >
              <div>
                <p className={styles.eyebrow}>Your first analysis</p>
                <h2 id="start-title">Open a terminal. Start exploring.</h2>
                <p className={styles.sectionLead}>
                  Install the plugins for your model and target, discover the
                  available backends, then run your first analysis.
                </p>
                <div className={styles.textLinks}>
                  <SiteLink to="https://github.com/arm/mlia#installation">
                    Installation guide
                  </SiteLink>
                  <SiteLink to="/api">Use the Python API</SiteLink>
                </div>
              </div>
              <SiteLink
                className={styles.secondaryButton}
                to="/cli#common-workflows"
              >
                Follow a workflow
              </SiteLink>
            </section>

            <section
              className={styles.pluginSection}
              aria-labelledby="plugins-title"
            >
              <div className={styles.sectionHeader}>
                <div>
                  <p className={styles.eyebrow}>The MLIA ecosystem</p>
                  <h2 id="plugins-title">Connect your toolkit.</h2>
                </div>
                <p>
                  The core provides shared workflows. Choose plugins for your
                  targets, converters, and visualization.
                </p>
              </div>
              <div className={styles.pluginGrid}>
                {plugins.map((plugin) => (
                  <SiteLink
                    className={styles.pluginCard}
                    indicator={false}
                    to={plugin.to}
                    key={plugin.name}
                  >
                    <span className={styles.tag}>{plugin.kind}</span>
                    <h3>{plugin.name}</h3>
                    <p>{plugin.description}</p>
                    <span className={styles.cardLink}>
                      Read the guide
                      <LinkIndicator to={plugin.to} />
                    </span>
                  </SiteLink>
                ))}
              </div>
            </section>
          </div>

          <section
            className={styles.contribute}
            aria-labelledby="contribute-title"
          >
            <div>
              <p className={styles.eyebrow}>Under the hood</p>
              <h2 id="contribute-title">
                Follow the analysis. Build what’s next.
              </h2>
              <p>
                Explore the architecture, follow an analysis, or build your own
                plugin.
              </p>
            </div>
            <div className={styles.actions}>
              <SiteLink
                className={styles.secondaryButton}
                to="/high_level_architecture"
              >
                Architecture
              </SiteLink>
              <SiteLink className={styles.secondaryButton} to="/development">
                Development
              </SiteLink>
            </div>
          </section>
        </div>
      </main>
    </Layout>
  );
}
