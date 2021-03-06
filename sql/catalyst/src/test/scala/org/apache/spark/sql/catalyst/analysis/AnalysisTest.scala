/*
 * Licensed to the Apache Software Foundation (ASF) under one or more
 * contributor license agreements.  See the NOTICE file distributed with
 * this work for additional information regarding copyright ownership.
 * The ASF licenses this file to You under the Apache License, Version 2.0
 * (the "License"); you may not use this file except in compliance with
 * the License.  You may obtain a copy of the License at
 *
 *    http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package org.apache.spark.sql.catalyst.analysis

import java.net.URI
import java.util.Locale

import org.apache.spark.sql.AnalysisException
import org.apache.spark.sql.catalyst.QueryPlanningTracker
import org.apache.spark.sql.catalyst.catalog.{CatalogDatabase, InMemoryCatalog, SessionCatalog}
import org.apache.spark.sql.catalyst.parser.ParseException
import org.apache.spark.sql.catalyst.plans.PlanTest
import org.apache.spark.sql.catalyst.plans.logical._
import org.apache.spark.sql.catalyst.rules.Rule
import org.apache.spark.sql.internal.SQLConf

trait AnalysisTest extends PlanTest {

  protected def extendedAnalysisRules: Seq[Rule[LogicalPlan]] = Nil

  protected def getAnalyzer: Analyzer = {
    val catalog = new SessionCatalog(new InMemoryCatalog, FunctionRegistry.builtin)
    catalog.createDatabase(
      CatalogDatabase("default", "", new URI("loc"), Map.empty),
      ignoreIfExists = false)
    catalog.createTempView("TaBlE", TestRelations.testRelation, overrideIfExists = true)
    catalog.createTempView("TaBlE2", TestRelations.testRelation2, overrideIfExists = true)
    catalog.createTempView("TaBlE3", TestRelations.testRelation3, overrideIfExists = true)
    catalog.createGlobalTempView("TaBlE4", TestRelations.testRelation4, overrideIfExists = true)
    catalog.createGlobalTempView("TaBlE5", TestRelations.testRelation5, overrideIfExists = true)
    new Analyzer(catalog) {
      override val extendedResolutionRules = EliminateSubqueryAliases +: extendedAnalysisRules
    }
  }

  protected def checkAnalysis(
      inputPlan: LogicalPlan,
      expectedPlan: LogicalPlan,
      caseSensitive: Boolean = true): Unit = {
    withSQLConf(SQLConf.CASE_SENSITIVE.key -> caseSensitive.toString) {
      val analyzer = getAnalyzer
      val actualPlan = analyzer.executeAndCheck(inputPlan, new QueryPlanningTracker)
      comparePlans(actualPlan, expectedPlan)
    }
  }

  protected def checkAnalysisWithoutViewWrapper(
      inputPlan: LogicalPlan,
      expectedPlan: LogicalPlan,
      caseSensitive: Boolean = true): Unit = {
    withSQLConf(SQLConf.CASE_SENSITIVE.key -> caseSensitive.toString) {
      val actualPlan = getAnalyzer.executeAndCheck(inputPlan, new QueryPlanningTracker)
      val transformed = actualPlan transformUp {
        case v: View if v.isTempViewStoringAnalyzedPlan => v.child
      }
      comparePlans(transformed, expectedPlan)
    }
  }

  protected override def comparePlans(
      plan1: LogicalPlan,
      plan2: LogicalPlan,
      checkAnalysis: Boolean = false): Unit = {
    // Analysis tests may have not been fully resolved, so skip checkAnalysis.
    super.comparePlans(plan1, plan2, checkAnalysis)
  }

  protected def assertAnalysisSuccess(
      inputPlan: LogicalPlan,
      caseSensitive: Boolean = true): Unit = {
    withSQLConf(SQLConf.CASE_SENSITIVE.key -> caseSensitive.toString) {
      val analyzer = getAnalyzer
      val analysisAttempt = analyzer.execute(inputPlan)
      try analyzer.checkAnalysis(analysisAttempt) catch {
        case a: AnalysisException =>
          fail(
            s"""
              |Failed to Analyze Plan
              |$inputPlan
              |
              |Partial Analysis
              |$analysisAttempt
            """.stripMargin, a)
      }
    }
  }

  protected def assertAnalysisError(
      inputPlan: LogicalPlan,
      expectedErrors: Seq[String],
      caseSensitive: Boolean = true): Unit = {
    withSQLConf(SQLConf.CASE_SENSITIVE.key -> caseSensitive.toString) {
      val analyzer = getAnalyzer
      val e = intercept[AnalysisException] {
        analyzer.checkAnalysis(analyzer.execute(inputPlan))
      }

      if (!expectedErrors.map(_.toLowerCase(Locale.ROOT)).forall(
          e.getMessage.toLowerCase(Locale.ROOT).contains)) {
        fail(
          s"""Exception message should contain the following substrings:
             |
             |  ${expectedErrors.mkString("\n  ")}
             |
             |Actual exception message:
             |
             |  ${e.getMessage}
           """.stripMargin)
      }
    }
  }

  protected def interceptParseException(
      parser: String => Any)(sqlCommand: String, messages: String*): Unit = {
    val e = intercept[ParseException](parser(sqlCommand))
    messages.foreach { message =>
      assert(e.message.contains(message))
    }
  }
}
