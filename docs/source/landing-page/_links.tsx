// SPDX-FileCopyrightText: Copyright 2026, Arm Limited and/or its affiliates.
// SPDX-License-Identifier: Apache-2.0

import Link from "@docusaurus/Link";
import type { ComponentProps, ReactElement } from "react";
import styles from "./index.module.css";

function isExternal(destination: string): boolean {
  return /^https?:\/\//i.test(destination);
}

export function LinkIndicator({ to }: { to: string }): ReactElement {
  const external = isExternal(to);
  return (
    <>
      <span
        className={styles.linkIndicator}
        aria-hidden="true"
        title={external ? "Opens in a new tab" : undefined}
      >
        {external ? "↗" : "→"}
      </span>
      {external && <span className={styles.srOnly}> (opens in a new tab)</span>}
    </>
  );
}

type Props = ComponentProps<typeof Link> & { indicator?: boolean };

export default function SiteLink({
  children,
  indicator = true,
  ...props
}: Props): ReactElement {
  const destination =
    typeof props.to === "string" ? props.to : String(props.href ?? "");
  const external = isExternal(destination);
  return (
    <Link
      {...props}
      target={external ? "_blank" : props.target}
      rel={external ? "noopener noreferrer" : props.rel}
    >
      {children}
      {indicator && <LinkIndicator to={destination} />}
    </Link>
  );
}
