# Parts of this class are from the Scanpy equivalent, see license below

# BSD 3-Clause License

# Copyright (c) 2017 F. Alexander Wolf, P. Angerer, Theis Lab
# All rights reserved.

# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:

# * Redistributions of source code must retain the above copyright notice, this
#   list of conditions and the following disclaimer.

# * Redistributions in binary form must reproduce the above copyright notice,
#   this list of conditions and the following disclaimer in the documentation
#   and/or other materials provided with the distribution.

# * Neither the name of the copyright holder nor the names of its
#   contributors may be used to endorse or promote products derived from
#   this software without specific prior written permission.

# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

# Further parts of this are from lamin-utils, see license below

#                                 Apache License
#                           Version 2.0, January 2004
#                        http://www.apache.org/licenses/
#
#   TERMS AND CONDITIONS FOR USE, REPRODUCTION, AND DISTRIBUTION
#
#   1. Definitions.
#
#      "License" shall mean the terms and conditions for use, reproduction,
#      and distribution as defined by Sections 1 through 9 of this document.
#
#      "Licensor" shall mean the copyright owner or entity authorized by
#      the copyright owner that is granting the License.
#
#      "Legal Entity" shall mean the union of the acting entity and all
#      other entities that control, are controlled by, or are under common
#      control with that entity. For the purposes of this definition,
#      "control" means (i) the power, direct or indirect, to cause the
#      direction or management of such entity, whether by contract or
#      otherwise, or (ii) ownership of fifty percent (50%) or more of the
#      outstanding shares, or (iii) beneficial ownership of such entity.
#
#      "You" (or "Your") shall mean an individual or Legal Entity
#      exercising permissions granted by this License.
#
#      "Source" form shall mean the preferred form for making modifications,
#      including but not limited to software source code, documentation
#      source, and configuration files.
#
#      "Object" form shall mean any form resulting from mechanical
#      transformation or translation of a Source form, including but
#      not limited to compiled object code, generated documentation,
#      and conversions to other media types.
#
#      "Work" shall mean the work of authorship, whether in Source or
#      Object form, made available under the License, as indicated by a
#      copyright notice that is included in or attached to the work
#      (an example is provided in the Appendix below).
#
#      "Derivative Works" shall mean any work, whether in Source or Object
#      form, that is based on (or derived from) the Work and for which the
#      editorial revisions, annotations, elaborations, or other modifications
#      represent, as a whole, an original work of authorship. For the purposes
#      of this License, Derivative Works shall not include works that remain
#      separable from, or merely link (or bind by name) to the interfaces of,
#      the Work and Derivative Works thereof.
#
#      "Contribution" shall mean any work of authorship, including
#      the original version of the Work and any modifications or additions
#      to that Work or Derivative Works thereof, that is intentionally
#      submitted to Licensor for inclusion in the Work by the copyright owner
#      or by an individual or Legal Entity authorized to submit on behalf of
#      the copyright owner. For the purposes of this definition, "submitted"
#      means any form of electronic, verbal, or written communication sent
#      to the Licensor or its representatives, including but not limited to
#      communication on electronic mailing lists, source code control systems,
#      and issue tracking systems that are managed by, or on behalf of, the
#      Licensor for the purpose of discussing and improving the Work, but
#      excluding communication that is conspicuously marked or otherwise
#      designated in writing by the copyright owner as "Not a Contribution."
#
#      "Contributor" shall mean Licensor and any individual or Legal Entity
#      on behalf of whom a Contribution has been received by Licensor and
#      subsequently incorporated within the Work.
#
#   2. Grant of Copyright License. Subject to the terms and conditions of
#      this License, each Contributor hereby grants to You a perpetual,
#      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#      copyright license to reproduce, prepare Derivative Works of,
#      publicly display, publicly perform, sublicense, and distribute the
#      Work and such Derivative Works in Source or Object form.
#
#   3. Grant of Patent License. Subject to the terms and conditions of
#      this License, each Contributor hereby grants to You a perpetual,
#      worldwide, non-exclusive, no-charge, royalty-free, irrevocable
#      (except as stated in this section) patent license to make, have made,
#      use, offer to sell, sell, import, and otherwise transfer the Work,
#      where such license applies only to those patent claims licensable
#      by such Contributor that are necessarily infringed by their
#      Contribution(s) alone or by combination of their Contribution(s)
#      with the Work to which such Contribution(s) was submitted. If You
#      institute patent litigation against any entity (including a
#      cross-claim or counterclaim in a lawsuit) alleging that the Work
#      or a Contribution incorporated within the Work constitutes direct
#      or contributory patent infringement, then any patent licenses
#      granted to You under this License for that Work shall terminate
#      as of the date such litigation is filed.
#
#   4. Redistribution. You may reproduce and distribute copies of the
#      Work or Derivative Works thereof in any medium, with or without
#      modifications, and in Source or Object form, provided that You
#      meet the following conditions:
#
#      (a) You must give any other recipients of the Work or
#          Derivative Works a copy of this License; and
#
#      (b) You must cause any modified files to carry prominent notices
#          stating that You changed the files; and
#
#      (c) You must retain, in the Source form of any Derivative Works
#          that You distribute, all copyright, patent, trademark, and
#          attribution notices from the Source form of the Work,
#          excluding those notices that do not pertain to any part of
#          the Derivative Works; and
#
#      (d) If the Work includes a "NOTICE" text file as part of its
#          distribution, then any Derivative Works that You distribute must
#          include a readable copy of the attribution notices contained
#          within such NOTICE file, excluding those notices that do not
#          pertain to any part of the Derivative Works, in at least one
#          of the following places: within a NOTICE text file distributed
#          as part of the Derivative Works; within the Source form or
#          documentation, if provided along with the Derivative Works; or,
#          within a display generated by the Derivative Works, if and
#          wherever such third-party notices normally appear. The contents
#          of the NOTICE file are for informational purposes only and
#          do not modify the License. You may add Your own attribution
#          notices within Derivative Works that You distribute, alongside
#          or as an addendum to the NOTICE text from the Work, provided
#          that such additional attribution notices cannot be construed
#          as modifying the License.
#
#      You may add Your own copyright statement to Your modifications and
#      may provide additional or different license terms and conditions
#      for use, reproduction, or distribution of Your modifications, or
#      for any such Derivative Works as a whole, provided Your use,
#      reproduction, and distribution of the Work otherwise complies with
#      the conditions stated in this License.
#
#   5. Submission of Contributions. Unless You explicitly state otherwise,
#      any Contribution intentionally submitted for inclusion in the Work
#      by You to the Licensor shall be under the terms and conditions of
#      this License, without any additional terms or conditions.
#      Notwithstanding the above, nothing herein shall supersede or modify
#      the terms of any separate license agreement you may have executed
#      with Licensor regarding such Contributions.
#
#   6. Trademarks. This License does not grant permission to use the trade
#      names, trademarks, service marks, or product names of the Licensor,
#      except as required for reasonable and customary use in describing the
#      origin of the Work and reproducing the content of the NOTICE file.
#
#   7. Disclaimer of Warranty. Unless required by applicable law or
#      agreed to in writing, Licensor provides the Work (and each
#      Contributor provides its Contributions) on an "AS IS" BASIS,
#      WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
#      implied, including, without limitation, any warranties or conditions
#      of TITLE, NON-INFRINGEMENT, MERCHANTABILITY, or FITNESS FOR A
#      PARTICULAR PURPOSE. You are solely responsible for determining the
#      appropriateness of using or redistributing the Work and assume any
#      risks associated with Your exercise of permissions under this License.
#
#   8. Limitation of Liability. In no event and under no legal theory,
#      whether in tort (including negligence), contract, or otherwise,
#      unless required by applicable law (such as deliberate and grossly
#      negligent acts) or agreed to in writing, shall any Contributor be
#      liable to You for damages, including any direct, indirect, special,
#      incidental, or consequential damages of any character arising as a
#      result of this License or out of the use or inability to use the
#      Work (including but not limited to damages for loss of goodwill,
#      work stoppage, computer failure or malfunction, or any and all
#      other commercial damages or losses), even if such Contributor
#      has been advised of the possibility of such damages.
#
#   9. Accepting Warranty or Additional Liability. While redistributing
#      the Work or Derivative Works thereof, You may choose to offer,
#      and charge a fee for, acceptance of support, warranty, indemnity,
#      or other liability obligations and/or rights consistent with this
#      License. However, in accepting such obligations, You may act only
#      on Your own behalf and on Your sole responsibility, not on behalf
#      of any other Contributor, and only if You agree to indemnify,
#      defend, and hold each Contributor harmless for any liability
#      incurred by, or claims asserted against, such Contributor by reason
#      of your accepting any such warranty or additional liability.
#
#   END OF TERMS AND CONDITIONS
#
#   APPENDIX: How to apply the Apache License to your work.
#
#      To apply the Apache License to your work, attach the following
#      boilerplate notice, with the fields enclosed by brackets "[]"
#      replaced with your own identifying information. (Don't include
#      the brackets!)  The text should be enclosed in the appropriate
#      comment syntax for the file format. We also recommend that a
#      file or class name and description of purpose be included on the
#      same "printed page" as the copyright notice for easier
#      identification within third-party archives.
#
#   Copyright [yyyy] [name of copyright owner]
#
#   Licensed under the Apache License, Version 2.0 (the "License");
#   you may not use this file except in compliance with the License.
#   You may obtain a copy of the License at
#
#       http://www.apache.org/licenses/LICENSE-2.0
#
#   Unless required by applicable law or agreed to in writing, software
#   distributed under the License is distributed on an "AS IS" BASIS,
#   WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#   See the License for the specific language governing permissions and
#   limitations under the License.
"""Logging and Profiling."""

import logging

# import platform
import sys
from datetime import UTC, datetime, timedelta, timezone
from logging import CRITICAL, DEBUG, ERROR, INFO, WARNING, getLevelName
from typing import Optional

# sys.stdout inside jupyter doesn't have reconfigure
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(errors="backslashreplace")  # type: ignore


HINT = 15
SAVE = 21
SUCCESS = 25
PRINT = 41  # always print
IMPORTANT = 31  # at warning level
IMPORTANT_HINT = 32  # at warning level
logging.addLevelName(HINT, "HINT")
logging.addLevelName(SAVE, "SAVE")
logging.addLevelName(SUCCESS, "SUCCESS")
logging.addLevelName(PRINT, "PRINT")
logging.addLevelName(IMPORTANT, "IMPORTANT")
logging.addLevelName(IMPORTANT_HINT, "IMPORTANT_HINT")


VERBOSITY_TO_LOGLEVEL = {
    0: "ERROR",  # 40
    1: "WARNING",  # 30
    2: "SUCCESS",  # 25
    3: "INFO",  # 20
    4: "HINT",  # 15
    5: "DEBUG",  # 10
}


LEVEL_TO_ICONS = {
    40: "✗",  # error
    32: "•",  # important hint
    31: "→",  # important
    30: "!",  # warning
    25: "✓",  # success
    21: "✓",  # save
    20: "•",  # info
    15: "•",  # hint
    10: "•",  # debug
}

# Add color codes
LEVEL_TO_COLORS = {
    40: "\033[91m",  # Red for error
    32: "\033[94m",  # Blue for important hint
    31: "\033[92m",  # Green for important
    30: "\033[93m",  # Yellow for warning
    25: "\033[92m",  # Green for success
    21: "\033[92m",  # Green for save
    20: "\033[94m",  # Blue for info
    15: "\033[96m",  # Cyan for hint
    10: "\033[90m",  # Grey for debug
}

RESET_COLOR = "\033[0m"


class RootLogger(logging.RootLogger):
    def __init__(self, level="INFO"):
        super().__init__(level)
        self.propagate = False
        self._verbosity: int = 1
        self.indent = ""
        RootLogger.manager = logging.Manager(self)

    def log(  # type: ignore
        self,
        level: int,
        msg: str,
        *,
        extra: dict | None = None,
        time: datetime = None,
        deep: str | None = None,
    ) -> datetime:
        """Log message with level and return current time.

        Args:
            level: Logging level.
            msg: Message to display.
            time: A time in the past. If this is passed, the time difference from then
                to now is appended to `msg` as ` (HH:MM:SS)`.
                If `msg` contains `{time_passed}`, the time difference is instead
                inserted at that position.
            deep: If the current verbosity is higher than the log function’s level,
                this gets displayed as well
            extra: Additional values you can specify in `msg` like `{time_passed}`.
        """
        now = datetime.now(UTC)
        time_passed: timedelta = None if time is None else now - time  # type: ignore
        extra = {
            **(extra or {}),
            "deep": (deep if getLevelName(VERBOSITY_TO_LOGLEVEL[self._verbosity]) < level else None),
            "time_passed": time_passed,
        }
        msg = f"{self.indent}{msg}"
        super().log(level, msg, extra=extra)
        return now

    def critical(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(CRITICAL, msg, time=time, deep=deep, extra=extra)

    def error(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(ERROR, msg, time=time, deep=deep, extra=extra)

    def warning(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(WARNING, msg, time=time, deep=deep, extra=extra)

    def important(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(IMPORTANT, msg, time=time, deep=deep, extra=extra)

    def important_hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(IMPORTANT_HINT, msg, time=time, deep=deep, extra=extra)

    def success(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(SUCCESS, msg, time=time, deep=deep, extra=extra)

    def info(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(INFO, msg, time=time, deep=deep, extra=extra)

    def save(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(SAVE, msg, time=time, deep=deep, extra=extra)

    def hint(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(HINT, msg, time=time, deep=deep, extra=extra)

    def debug(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(DEBUG, msg, time=time, deep=deep, extra=extra)

    def print(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(PRINT, msg, time=time, deep=deep, extra=extra)

    # backward compat
    def download(self, msg, *, time=None, deep=None, extra=None) -> datetime:  # type: ignore
        return self.log(SAVE, msg, time=time, deep=deep, extra=extra)


class _LogFormatter(logging.Formatter):
    def __init__(self, fmt="{levelname}: {message}", datefmt="%Y-%m-%d %H:%M", style="{"):
        super().__init__(fmt, datefmt, style)

    def base_format(self, record: logging.LogRecord):
        # if platform.system() == "Windows":
        #     return f"{record.levelname}:" + " {message}"
        # else:
        if LEVEL_TO_ICONS.get(record.levelno) is not None:
            color = LEVEL_TO_COLORS.get(record.levelno, "")
            icon = LEVEL_TO_ICONS[record.levelno]
            return f"{color}{icon}{RESET_COLOR}" + " {message}"
        else:
            return "{message}"

    def format(self, record: logging.LogRecord):
        format_orig = self._style._fmt
        self._style._fmt = self.base_format(record)
        if record.time_passed:  # type: ignore
            if "{time_passed}" in record.msg:
                record.msg = record.msg.replace(
                    "{time_passed}",
                    record.time_passed,  # type: ignore
                )
            else:
                self._style._fmt += " ({time_passed})"
        if record.deep:  # type: ignore
            record.msg = f"{record.msg}: {record.deep}"  # type: ignore
        result = logging.Formatter.format(self, record)
        self._style._fmt = format_orig
        return result


logger = RootLogger()


def set_handler(logger):
    h = logging.StreamHandler(stream=sys.stdout)
    h.setFormatter(_LogFormatter())
    h.setLevel(logger.level)
    if len(logger.handlers) == 1:
        logger.removeHandler(logger.handlers[0])
    elif len(logger.handlers) > 1:
        raise RuntimeError("Lamin's root logger somehow got more than one handler")
    logger.addHandler(h)


set_handler(logger)


def set_log_level(logger, level: int):
    logger.setLevel(level)
    (h,) = logger.handlers  # can only be 1
    h.setLevel(level)


# this also sets it for the handler
RootLogger.set_level = set_log_level  # type: ignore


def set_verbosity(logger, verbosity: int):
    if verbosity not in VERBOSITY_TO_LOGLEVEL:
        raise ValueError(f"verbosity needs to be one of {set(VERBOSITY_TO_LOGLEVEL.keys())}")
    logger.set_level(VERBOSITY_TO_LOGLEVEL[verbosity])
    logger._verbosity = verbosity


RootLogger.set_verbosity = set_verbosity  # type: ignore


def mute(logger):
    """Context manager to mute logger."""

    class Muted:
        def __enter__(self):
            self.original_verbosity = logger._verbosity
            logger.set_verbosity(0)

        def __exit__(self, exc_type, exc_val, exc_tb):
            logger.set_verbosity(self.original_verbosity)

    return Muted()


RootLogger.mute = mute  # type: ignore
